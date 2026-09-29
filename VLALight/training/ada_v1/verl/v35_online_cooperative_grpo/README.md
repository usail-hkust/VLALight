# V35 Online Cooperative GRPO

This directory contains the Stage 2 online path for `ada_v1`. Stage 1
perception remains an SFT/CE task; the online rollout consumes the resulting
local and cooperative JSON as structured text.

`rollout_city` treats one city snapshot as one synchronized decision point:
all intersections use the same `step`, all policy responses are collected,
signals are applied together, and the simulator advances a fixed number of
decision cycles (three by default). Six trials assign every intersection
exactly three `fast` and three `slow` modes. Results are kept per intersection
and include the independent `global_queue_reward`, `local_queue_reward`,
`reasoning_cost_reward`, and `format_penalty` fields expected by GDPO.

`flatten_for_gdpo` turns these nested results into individual records. Its
`group_id` is `city:step:intersection_id`, which groups the six candidates for
one target while retaining each target's independent prompt and response.
Configure the existing GDPO implementation with these component keys:

```yaml
algorithm:
  adv_estimator: gdpo
  gdpo_reward_keys:
    - global_queue_reward
    - local_queue_reward
    - reasoning_cost_reward
    - format_penalty
  gdpo_reward_weights: [1.0, 1.0, 1.0, 1.0]
```

The implementation depends only on the small `SimulatorAdapter` protocol, so a
SUMO wrapper can implement `restore`, `apply_signals`, `advance`, and
`queue_metrics` without coupling the reward code to a particular environment.

`sumo_adapter.py` provides that wrapper for the repository's existing
`utils.sumo_env.SUMOEnv`.  Create one actor-local adapter with
`SUMOEnvFactory(config_by_city, paths_by_city, work_root, repo_root=...)`.
The adapter accepts prompt phase names (`ETWT`, `NTST`, `ELWL`, `NLSL`), maps
them to each intersection's `control_phases` index, requires one signal for
every active intersection, and submits the complete table in one
`SUMOEnv.step` call.  `decision_cycles=3` advances `90` seconds by default

Use `rollout_one_assignment_temporal` for the online t0/t1/t2/t3 path. With
`decision_cycles=3`, it performs four decisions: t0 optionally invokes the
injected `stage1_fn` to produce only each intersection's local `<perception>`
block; `router_fn` assembles the canonical `{local, neighbors}` object consumed
by Stage 2. At t1-t3, `observation_fn` rebuilds snapshots from SUMO and
`router_fn` is applied again, while only Stage 2 runs. It advances SUMO one
cycle between decisions and exposes trajectory-level sums in
`cumulative_rewards` for GDPO. The legacy rollout is unchanged.
(three 30-second decision cycles), while `queue_metrics()` returns total
incoming waiting vehicles per intersection.

Typical integration:

```python
from v35_online_cooperative_grpo import SUMOEnvFactory, rollout_city

factory = SUMOEnvFactory(config_by_city, paths_by_city, work_root,
                         repo_root=repo_root)
simulator = factory(city, seed, actor_id="rollout_0")
snapshot_path = simulator.save_snapshot(snapshot_dir / "t0.xml")
# Build a CitySnapshot at this same simulator time, then:
results = rollout_city(city_snapshot, simulator, policy_fn)
simulator.close()
```

For six counterfactual trials, pass the same saved snapshot path in
`CitySnapshot.simulator_snapshot`; `rollout_city` restores it before each
trial and only applies actions after every intersection response has arrived.
The coordinator rejects multi-rollout calls without a snapshot because those
would compare different simulator states.

Run the CPU contract check from the `verl` directory:

```text
python v35_online_cooperative_grpo/verify_online_cooperative.py
```

## Ray process-level execution

`ray_rollout.py` is the process-level coordinator. For each input city sample,
create one `SUMOMasterActor`; it owns the master SUMO and stays at the same
`city/step` while candidates run. `RayCityRolloutCoordinator` then creates six
`SUMORolloutActor` instances. Each rollout actor starts its own SUMO process,
loads a private copy of the master's snapshot, collects all intersections'
responses, applies the complete signal table once, and advances three cycles.
The six actors therefore represent six counterfactual worlds rather than six
threads operating on one simulator.

```python
coordinator = RayCityRolloutCoordinator(env_factory, num_rollouts=6)
master = coordinator.create_master_actor(city, seed, actor_id=f"master_{sample_id}")
results = coordinator.run_city_rollouts_from_master_actor(
    snapshot, master, policy_factory, snapshot_dir,
)
ray.kill(master, no_restart=True)
```

For a batch of four slots, create four persistent master actors (one per slot).
Each master produces six rollout actors, so the maximum SUMO process count is
`4 * (1 + 6)` while those samples are evaluated. Use Ray resource limits or
batch the slots if the host cannot support that many SUMO processes. After a
candidate is selected, commit it to the same master slot; the next batch call
then reads the next decision time from that master. The snapshot directory
must be shared by the master and rollout workers.

## VERL batch bridge

`verl_adapter.py` keeps this online path separate from the generic agent loop.
The dataloader rows only need the metadata emitted by
`OnlineSampleSpec.as_dict()`. A trainer-side hook can pass a sampled
`DataProto` to `OnlineVERLCollector`:

```python
from v35_online_cooperative_grpo import OnlineVERLCollector

collector = OnlineVERLCollector(runtime, config=config, split="train")
online_batch = collector.collect(fake_batch)
records = online_batch.records
```

`collect` uses `city`, `stream_id`, `seed`, and `ordinal` from each row, reads
a same-step topology-complete snapshot from each persistent master, runs the
six synchronized candidates, commits the selected complete signal table, and
returns one record per intersection and candidate. The record `uid` is
`city:step:intersection_id`, so the six records for one target form one GDPO
group. Each record includes global/local queue rewards, reasoning cost, format
penalty, forced mode, and before/after queue maps. `extra_info.selected` marks
the candidate committed to the master.

If token tensors are needed immediately, call
`collector.collect(fake_batch, tokenizer=tokenizer, materialize=True)`. This
creates a text-only `DataProto` and never adds image/video placeholders. The
generic VERL trainer still owns policy generation and optimizer updates; the
collector is the explicit online data/reward boundary and should be called
where the trainer currently obtains its rollout batch. The
`policy_factory` passed to `OnlineCooperativeRuntime` must be connected to the
current policy/LLM client by the training entrypoint; this module does not load
another model or silently replace VERL's agent loop.
