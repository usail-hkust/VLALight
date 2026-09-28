"""
Tool Executor for VLM Traffic Signal Control.

Loads tool definitions from JSON files, executes tools in a sandboxed manner
(subprocess isolation, inspired by code_sandbox.py), and formats results
for the VLM API.

Usage:
    executor = ToolExecutor(scenario="jinan", tools_dir="tools/tsc_tools")
    # Get OpenAI-compatible tool definitions for VLM API
    tool_defs = executor.get_tool_definitions()
    # Execute a tool call returned by VLM
    result = executor.execute(tool_call, context)
"""

import os
import sys
import json
import base64
import importlib
import traceback
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional


class ToolExecutor:
    """
    Sandboxed tool executor that:
    1. Loads tool definitions from tools/tsc_tools/*.json
    2. Provides OpenAI-compatible tool schemas for VLM API calls
    3. Executes tool functions in isolated subprocess (sandbox mode)
       or in-process (fast mode) based on tool config
    4. Formats results (including images) for VLM consumption
    """

    def __init__(self,
                 scenario: str = "jinan",
                 tools_dir: Optional[str] = None,
                 log_dir: Optional[str] = None,
                 sandbox_dir: Optional[str] = None):
        """
        Args:
            scenario: Traffic scenario name (jinan/hangzhou/newyork)
            tools_dir: Path to tool definition directory. Defaults to
                       <project_root>/tools/tsc_tools/
            log_dir: Directory for tool execution logs/outputs (fallback)
            sandbox_dir: Dedicated directory for tool sandbox artifacts.
                         Defaults to {log_dir}/tool_sandbox/ if not specified.
        """
        self.scenario = scenario
        self.project_root = Path(__file__).resolve().parent.parent

        if tools_dir is None:
            self.tools_dir = self.project_root / "tools" / "tsc_tools"
        else:
            self.tools_dir = Path(tools_dir)

        self.log_dir = Path(log_dir) if log_dir else None
        if sandbox_dir:
            self.sandbox_dir = Path(sandbox_dir)
        elif self.log_dir:
            self.sandbox_dir = self.log_dir / "tool_sandbox"
        else:
            self.sandbox_dir = None

        # Load registry and tool definitions
        self._registry: Dict[str, Any] = {}
        self._tool_defs: Dict[str, Dict] = {}
        self._load_registry()

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def get_tool_definitions(self) -> List[Dict]:
        """
        Return OpenAI-compatible tool definitions for the VLM API `tools` param.

        Returns:
            List of dicts, each with {"type": "function", "function": {...}}
        """
        definitions = []
        for name, tool_def in self._tool_defs.items():
            definitions.append({
                "type": "function",
                "function": {
                    "name": tool_def["function"]["name"],
                    "description": tool_def["function"]["description"],
                    "parameters": tool_def["function"]["parameters"],
                }
            })
        return definitions

    def execute(self,
                tool_name: str,
                arguments: Dict[str, Any],
                context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool by name with given arguments.

        Args:
            tool_name: Name of the tool (e.g. "lane_segmentation")
            arguments: Tool arguments from VLM (e.g. {"direction": "S"})
            context: Execution context containing:
                - image_paths_dict: {direction: image_path}
                - tls_id: intersection ID
                - step_num: current step number
                - log_dir: output directory for tool artifacts

        Returns:
            Dict with keys:
                - success: bool
                - result: tool-specific result dict (e.g. lane info)
                - images: list of image file paths to append to VLM
                - error: error message if failed
                - summary: human-readable summary for VLM text
        """
        if tool_name not in self._tool_defs:
            return {
                "success": False,
                "result": None,
                "images": [],
                "error": f"Unknown tool: {tool_name}",
                "summary": f"Error: tool '{tool_name}' not found.",
            }

        tool_def = self._tool_defs[tool_name]
        executor_cfg = tool_def.get("executor", {})
        use_sandbox = executor_cfg.get("sandbox", False)
        timeout = executor_cfg.get("timeout", 30)

        try:
            if use_sandbox:
                raw_result = self._execute_in_subprocess(
                    tool_name, tool_def, arguments, context, timeout)
            else:
                raw_result = self._execute_in_process(
                    tool_name, tool_def, arguments, context)

            # Post-process: collect image paths from result
            images = self._collect_images(raw_result)
            summary = self._build_summary(tool_name, arguments, raw_result)

            return {
                "success": True,
                "result": raw_result,
                "images": images,
                "error": None,
                "summary": summary,
            }

        except Exception as e:
            error_msg = f"Tool '{tool_name}' execution failed: {e}"
            print(f"Warning: {error_msg}")
            traceback.print_exc()
            return {
                "success": False,
                "result": None,
                "images": [],
                "error": error_msg,
                "summary": f"Error executing {tool_name}: {e}",
            }

    def has_tools(self) -> bool:
        """Return True if any tools are loaded."""
        return len(self._tool_defs) > 0

    def list_tool_names(self) -> List[str]:
        """Return list of available tool names."""
        return list(self._tool_defs.keys())

    # ------------------------------------------------------------------ #
    #  Internal: Registry loading                                         #
    # ------------------------------------------------------------------ #

    def _load_registry(self):
        """Load tool_registry.json and all referenced tool definition files."""
        registry_path = self.tools_dir / "tool_registry.json"
        if not registry_path.exists():
            print(f"Warning: Tool registry not found at {registry_path}")
            return

        with open(registry_path, 'r', encoding='utf-8') as f:
            self._registry = json.load(f)

        for tool_entry in self._registry.get("tools", []):
            if not tool_entry.get("enabled", True):
                continue
            # Check scenario filter
            allowed_scenarios = tool_entry.get("scenarios", [])
            if allowed_scenarios and self.scenario not in allowed_scenarios:
                continue

            name = tool_entry["name"]
            def_file = self.tools_dir / tool_entry["definition_file"]
            if not def_file.exists():
                print(f"Warning: Tool definition not found: {def_file}")
                continue

            with open(def_file, 'r', encoding='utf-8') as f:
                self._tool_defs[name] = json.load(f)

    # ------------------------------------------------------------------ #
    #  Internal: In-process execution (fast, no isolation)                #
    # ------------------------------------------------------------------ #

    def _execute_in_process(self,
                            tool_name: str,
                            tool_def: Dict,
                            arguments: Dict[str, Any],
                            context: Dict[str, Any]) -> Dict:
        """Execute tool directly in current process (no sandbox)."""
        executor_cfg = tool_def["executor"]
        module_path = executor_cfg["module"]
        class_name = executor_cfg.get("class")
        method_name = executor_cfg.get("method")

        # Dynamic import
        mod = importlib.import_module(module_path)

        if class_name:
            cls = getattr(mod, class_name)
            # Build constructor args from arguments + context
            init_kwargs = self._build_init_kwargs(tool_name, arguments, context)
            instance = cls(**init_kwargs)
            func = getattr(instance, method_name)
        else:
            func = getattr(mod, method_name)

        # Build call args
        call_kwargs = self._build_call_kwargs(tool_name, arguments, context)
        return func(**call_kwargs)

    # ------------------------------------------------------------------ #
    #  Internal: Subprocess execution (sandboxed, inspired by             #
    #            code_sandbox.py)                                         #
    # ------------------------------------------------------------------ #

    def _execute_in_subprocess(self,
                               tool_name: str,
                               tool_def: Dict,
                               arguments: Dict[str, Any],
                               context: Dict[str, Any],
                               timeout: int = 30) -> Dict:
        """
        Execute tool in a subprocess for isolation.
        Serializes arguments to a temp JSON, runs a wrapper script,
        and reads the result JSON back.
        """
        executor_cfg = tool_def["executor"]
        module_path = executor_cfg["module"]
        class_name = executor_cfg.get("class")
        method_name = executor_cfg.get("method")

        # Prepare temp directory for this execution
        output_dir = self._get_output_dir(tool_name, context)
        os.makedirs(output_dir, exist_ok=True)

        # Serialize input
        input_data = {
            "module_path": module_path,
            "class_name": class_name,
            "method_name": method_name,
            "init_kwargs": self._build_init_kwargs(tool_name, arguments, context),
            "call_kwargs": self._build_call_kwargs(tool_name, arguments, context),
            "output_dir": output_dir,
        }

        input_file = os.path.join(output_dir, "_tool_input.json")
        result_file = os.path.join(output_dir, "_tool_result.json")

        with open(input_file, 'w', encoding='utf-8') as f:
            json.dump(input_data, f, ensure_ascii=False, indent=2)

        # Build wrapper script
        wrapper_code = self._build_wrapper_script(input_file, result_file)
        wrapper_file = os.path.join(output_dir, "_tool_wrapper.py")
        with open(wrapper_file, 'w', encoding='utf-8') as f:
            f.write(wrapper_code)

        # Execute in subprocess
        try:
            result = subprocess.run(
                [sys.executable, wrapper_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8',
                cwd=str(self.project_root),
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip() or result.stdout.strip() or "Unknown subprocess error"
                raise RuntimeError(
                    f"Subprocess exited with code {result.returncode}: {error_msg}")

            # Read result
            if not os.path.exists(result_file):
                raise RuntimeError(
                    f"Tool did not produce result file. stdout: {result.stdout[:500]}")

            with open(result_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Tool '{tool_name}' timed out ({timeout}s)")

    def _build_wrapper_script(self, input_file: str, result_file: str) -> str:
        """
        Generate a Python wrapper script that:
        1. Reads tool input from JSON
        2. Imports and executes the tool
        3. Writes result to JSON
        """
        project_root_str = str(self.project_root).replace('\\', '/')
        input_file_str = input_file.replace('\\', '/')
        result_file_str = result_file.replace('\\', '/')

        return f'''# -*- coding: utf-8 -*-
"""Auto-generated tool execution wrapper (sandboxed)."""
import sys
import json
import importlib
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, r"{project_root_str}")

def main():
    # Read input
    with open(r"{input_file_str}", "r", encoding="utf-8") as f:
        input_data = json.load(f)

    module_path = input_data["module_path"]
    class_name = input_data.get("class_name")
    method_name = input_data.get("method_name")
    init_kwargs = input_data.get("init_kwargs", {{}})
    call_kwargs = input_data.get("call_kwargs", {{}})

    # Import module
    mod = importlib.import_module(module_path)

    # Execute
    if class_name:
        cls = getattr(mod, class_name)
        instance = cls(**init_kwargs)
        func = getattr(instance, method_name)
    else:
        func = getattr(mod, method_name)

    result = func(**call_kwargs)

    # Write result
    with open(r"{result_file_str}", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Write error as result
        error_result = {{"error": str(e), "traceback": traceback.format_exc()}}
        with open(r"{result_file_str}", "w", encoding="utf-8") as f:
            json.dump(error_result, f, ensure_ascii=False, indent=2)
        sys.exit(1)
'''

    # ------------------------------------------------------------------ #
    #  Internal: Argument building (tool-specific logic)                  #
    # ------------------------------------------------------------------ #

    def _build_init_kwargs(self,
                           tool_name: str,
                           arguments: Dict[str, Any],
                           context: Dict[str, Any]) -> Dict:
        """Build constructor kwargs from tool definition's init_kwargs_map."""
        tool_def = self._tool_defs.get(tool_name, {})
        mapping = tool_def.get("executor", {}).get("init_kwargs_map", {})
        return self._resolve_kwargs_map(mapping, tool_name, arguments, context)

    def _build_call_kwargs(self,
                           tool_name: str,
                           arguments: Dict[str, Any],
                           context: Dict[str, Any]) -> Dict:
        """Build method call kwargs from tool definition's call_kwargs_map."""
        tool_def = self._tool_defs.get(tool_name, {})
        mapping = tool_def.get("executor", {}).get("call_kwargs_map", {})
        return self._resolve_kwargs_map(mapping, tool_name, arguments, context)

    def _resolve_kwargs_map(self,
                            mapping: Dict[str, Dict],
                            tool_name: str,
                            arguments: Dict[str, Any],
                            context: Dict[str, Any]) -> Dict:
        """Resolve a kwargs_map declaration into actual keyword arguments.

        Supported source types:
          - arguments: value from VLM tool call arguments
          - executor: value from ToolExecutor attributes (e.g. scenario)
          - context: value from execution context dict
          - context_lookup: lookup context[dict_key][arguments[index_key]]
                    - context_list_lookup: lookup context[list_key][arguments[index_key]]
          - output_dir: auto-generated sandbox output directory
        """
        result = {}
        for param_name, spec in mapping.items():
            source = spec.get("source", "")
            if source == "arguments":
                key = spec.get("key", param_name)
                result[param_name] = arguments.get(key, spec.get("default"))
            elif source == "executor":
                key = spec.get("key", param_name)
                result[param_name] = getattr(self, key, spec.get("default"))
            elif source == "context":
                key = spec.get("key", param_name)
                result[param_name] = context.get(key, spec.get("default"))
            elif source == "context_lookup":
                dict_key = spec.get("dict_key", "")
                index_key = spec.get("index_key", "")
                lookup_dict = context.get(dict_key, {})
                index_val = arguments.get(index_key, "")
                value = lookup_dict.get(index_val, spec.get("default", ""))
                if not value and param_name == "image_path":
                    available = list(lookup_dict.keys())
                    raise ValueError(
                        f"No image available for direction '{index_val}'. "
                        f"Available directions: {available}. "
                        f"Only call this tool with an available direction."
                    )
                result[param_name] = value
            elif source == "context_list_lookup":
                list_key = spec.get("list_key", "")
                index_key = spec.get("index_key", "")
                lookup_list = context.get(list_key, [])
                index_val = arguments.get(index_key, 0)
                try:
                    idx = int(index_val)
                except (TypeError, ValueError):
                    idx = 0
                if isinstance(lookup_list, list) and 0 <= idx < len(lookup_list):
                    result[param_name] = lookup_list[idx]
                else:
                    result[param_name] = spec.get("default", "")
            elif source == "output_dir":
                result[param_name] = self._get_output_dir(tool_name, context)
            else:
                result[param_name] = spec.get("default")
        return result

    # ------------------------------------------------------------------ #
    #  Internal: Result post-processing                                   #
    # ------------------------------------------------------------------ #

    def _collect_images(self, result: Dict) -> List[str]:
        """Extract image file paths from tool result.

        For lane_segmentation, only zone images (near/far) are collected,
        not the full lane crop, to avoid redundant VLM token consumption.
        """
        images = []
        for lane in result.get("lanes", []):
            # Skip full lane image — zone images already cover the same area
            # Collect zone images nested within each lane
            for zone in lane.get("zones", []):
                zpath = zone.get("zone_image_path", "")
                if zpath and os.path.exists(zpath):
                    images.append(zpath)
                elif zpath:
                    print(f"Warning: zone image not found: {zpath}")
        # Top-level zones (e.g. distance_zone_segmentation)
        for zone in result.get("zones", []):
            path = zone.get("zone_image_path", "")
            if path and os.path.exists(path):
                images.append(path)
            elif path:
                print(f"Warning: zone image not found: {path}")
        # Handle top-level image path fields (e.g. vehicle_detection returns
        # visualization_image_path directly on the result dict)
        for key in ("visualization_image_path", "image_path", "output_image_path"):
            path = result.get(key, "")
            if path and os.path.exists(path) and path not in images:
                images.append(path)
        return images

    def _build_summary(self,
                       tool_name: str,
                       arguments: Dict[str, Any],
                       result: Dict) -> str:
        """Build a human-readable summary of tool execution for VLM."""
        if tool_name == "lane_segmentation":
            direction = arguments.get("direction", "?")
            lanes = result.get("lanes", [])
            
            lines = [
                f"Lane segmentation completed for direction {direction}.",
                f"Segmented into {len(lanes)} lanes (each split into near/far zones):"
            ]
            for i, lane in enumerate(lanes):
                lane_dir = lane.get("lane_direction", "Unknown")
                lines.append(f"  Lane {i + 1}: {lane_dir}")
            
            lines.append(f"\nCount vehicles yourself from the returned lane/zone images."
                         f" Each lane has a near (queuing) and far (approaching) zone image."
                         f" SUM near+far counts for each lane to get L/T/R totals for <counts>."
                         f" Near zone = vehicles waiting at stop line; Far zone = vehicles approaching."
                         f"\nIMPORTANT: Image labels show ACTUAL traffic function (right-turn/straight/left-turn),"
                         f" NOT image-space position. Trust the labels.")

            return "\n".join(lines)

        if tool_name == "distance_zone_segmentation":
            direction = arguments.get("direction", "?")
            zones = result.get("zones", [])
            lines = [
                f"Distance-zone segmentation completed for direction {direction}.",
                f"Split into {len(zones)} zones:"
            ]
            for zone in zones:
                zone_name = zone.get("zone_name", zone.get("zone_key", "Unknown"))
                lines.append(f"  Zone {zone.get('zone_index', '?')}: {zone_name}")
            return "\n".join(lines)

        if tool_name == "congestion_analysis":
            text_report = result.get("text_report", "")
            if text_report:
                return text_report
            return "Congestion analysis completed (no report generated)."

        if tool_name == "queue_length_estimation":
            text_report = result.get("text_report", "")
            if text_report:
                return text_report
            q = result.get("queue_length_px", 0)
            mode = result.get("queue_tail_mode", "unknown")
            return f"Queue length estimation completed: {q}px (mode={mode})."

        return f"Tool '{tool_name}' executed successfully."

    def _get_output_dir(self, tool_name: str, context: Dict) -> str:
        """Get output directory for tool sandbox artifacts.

        Directory format:
            {sandbox_dir}/{tls_id}_step{N}_r{round}_c{call}_{tool_name}/

        Each tool invocation gets a unique sandbox directory so that
        multiple calls within the same Agent Loop step never collide.
        """
        base = self.sandbox_dir or self.log_dir or Path(context.get("log_dir", "/tmp"))
        tls_id = context.get("tls_id", "unknown")
        step_num = context.get("step_num", 0)
        round_idx = context.get("round_idx", 0)
        call_idx = context.get("call_idx", 0)
        output_dir = str(
            base / f"{tls_id}_step{step_num}_r{round_idx}_c{call_idx}_{tool_name}")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    # ------------------------------------------------------------------ #
    #  Static helpers for VLM message formatting                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def format_tool_result_message(tool_name: str,
                                   tool_result: Dict[str, Any]) -> Dict:
        """
        Format tool result as an OpenAI-compatible tool message.

        Returns:
            {"role": "tool", "tool_call_id": ..., "content": ...}
        """
        return {
            "role": "tool",
            "name": tool_name,
            "content": json.dumps({
                "success": tool_result["success"],
                "summary": tool_result.get("summary", ""),
                "error": tool_result.get("error"),
            }, ensure_ascii=False),
        }

    @staticmethod
    def encode_image_base64(image_path: str) -> Optional[Dict]:
        """
        Encode an image file to base64 for VLM API.

        Returns:
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
            or None if file doesn't exist.
        """
        if not os.path.exists(image_path):
            return None
        with open(image_path, 'rb') as f:
            data = f.read()
        b64 = base64.b64encode(data).decode('utf-8')
        ext = os.path.splitext(image_path)[1].lower()
        mime = 'image/jpeg' if ext in ('.jpg', '.jpeg') else 'image/png'
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"}
        }
