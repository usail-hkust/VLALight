'''
# Anonymous project source
@Date: 2024-07-13 20:53:01
@Description: ???????? ??? SUMO ????????panda3d
LastEditTime: 2025-07-28 21:13:10
'''
import math
from loguru import logger
from typing import Dict, List

from ..traffic_elements.vehicle import Vehicle3DElement
from ..traffic_elements.traffic_signals import TLS3DElement
from ..traffic_elements.aircraft import Aircraft3DElement
from ...vis3d_utils.core_math import calculate_center_point, vec_to_radians, vec_2d

VALID_SENSORS = {
    'aircraft': ['aircraft_all', 'aircraft_vehicle'],
    'tls': [
        'junction_front_all', 'junction_front_vehicle', 
        'junction_back_all', 'junction_back_vehicle'
    ],
    'vehicle': [
        'front_left_all', 'front_right_all', 'front_all', 
        'back_left_all', 'back_right_all', 'back_all',
        'front_left_vehicle', 'front_right_vehicle', 'front_vehicle', 
        'back_left_vehicle', 'back_right_vehicle', 'back_vehicle',
        'bev_all', 'bev_vehicle'
    ]
}


class SceneSync(object):
    def __init__(
            self, root_np, showbase_instance, 
            sensor_config:Dict[str, List[str]],
            preset:str='480P', resolution:float=1.0,
            vehicle_model:str='low',
            tls_batch_size:int=None,
            keep_batch_sensors:bool=True,
            reuse_batch_sensors:bool=False,
            step_task_manager:bool=True,
        ) -> None:
        """????????? object

        Args:
            root_np: ?????Node
            showbase_instance: Pnada3D ShowBase
            sensor_config (Dict[str, List[str]]): ??????????????????????
            preset (str, optional): ??????????????'720x480', '360x240' ???. Defaults to '480p'.
            resolution (float, optional): ???????????1.0 ???????? 0.5 ?????????. Defaults to 1.0.
            tls_batch_size (int, optional): ???????????????????? Defaults to None.
        """
        self.root_np = root_np
        self.showbase_instance = showbase_instance
        self.vehicle_model = vehicle_model # ??????????????3D ???
        self.sensor_config = sensor_config # ??? object ????????????
        
        # ???????????
        self.tls_batch_size = tls_batch_size
        self.keep_batch_sensors = keep_batch_sensors
        self.reuse_batch_sensors = reuse_batch_sensors
        self.step_task_manager = step_task_manager
        self._tls_init_obs = None  # ???????????tls ??????????????????
        self._all_tls_ids = []  # ?????????????tls_id ???
        self._current_batch_index = 0  # ?????????
        self._initialized_tls_ids = set()  # ????????tls_id ???

        # ??????????????????????????
        presets = {
            '320P': (320, 240),   # 320x240??? NTSC ?????
            '480P': (720, 480),   # 720x480?????480P??
            '720P': (1280, 720),  # 1280x720??D ?????
            '1080P': (1920, 1080) # 1920x1080??ull HD??
        }
        if preset not in presets:
            raise ValueError(f"Invalid preset: {preset}. Valid presets are: {list(presets.keys())}")
        self.fig_width, self.fig_height = presets.get(preset) # ????????????
        self.resolution = resolution

        if not SceneSync.validate_sensor_config(sensor_config):
            logger.info("SIM: Sensor configuration validation failed. Please check the errors above.")
            raise ValueError(f"?????????????? {sensor_config}")
            
        # ????????? element
        self._vehicle_elements = {} # ???????????
        self._tls_elements = {} # ????????(?????in road ???????, ?????????????? ??? sensor
        self._tls_element_pool = []
        self._aircraft_elements = {} # ??? aircraft ??????, ????????

    @staticmethod
    def validate_sensor_config(sensor_config: Dict[str, List[str]]) -> bool:
        """Validate the sensor configuration against the predefined valid sensors.
        """
        for category, object_sensors in sensor_config.items():
            # 1. ?????????????object ????????????, ???????? ?????vehicle, tls ??aircraft
            if category not in VALID_SENSORS: # ???????????? objects ????????
                logger.error(f"SIM: Invalid category: {category}. Valid categories are {list(VALID_SENSORS.keys())}.")
                return False
            
            # 2. ????????????????????????? ??? tls ??? junction_front_all 
            for object_id, sensors_info in object_sensors.items():            
                # ?????????????
                invalid_sensors = set(sensors_info.get('sensor_types', [])) - set(VALID_SENSORS[category])
                if invalid_sensors:
                    logger.error(f"SIM: Invalid sensors in {category}: {invalid_sensors}. Valid sensors are {VALID_SENSORS[category]}.")
                    return False
        return True

    def reset(self, tshub_init_obs) -> None:
        # ?????& ????????????????? sensors
        self.remove_missing_elements(set(), self._vehicle_elements, 'vehicle')
        self.remove_missing_elements(set(), self._aircraft_elements, 'aircraft')
        
        # ????????? (???????????& ??????????????& ???????????????)
        if (not self._tls_elements) and ("tls" in tshub_init_obs) and ("tls" in self.sensor_config):
            # ??? tls ???????????????????????
            self._tls_init_obs = tshub_init_obs['tls']
            # ????????????????tls_id
            self._all_tls_ids = [tls_id for tls_id in tshub_init_obs['tls'].keys() 
                                 if tls_id in self.sensor_config['tls']]
            self._current_batch_index = 0
            self._initialized_tls_ids = set()
            
            if self.tls_batch_size is None:
                # ?????????????????
                for tls_id in self._all_tls_ids:
                    self._initialize_tls_elements(tls_id, tshub_init_obs['tls'][tls_id])
                    self._initialized_tls_ids.add(tls_id)
                logger.info(f'SIM: ??????????????{len(self._all_tls_ids)} ?????????')
            else:
                # ????????????????????
                self._initialize_current_batch()
                total_batches = (len(self._all_tls_ids) + self.tls_batch_size - 1) // self.tls_batch_size
                logger.info(
                    f'SIM: batch sensor mode, total_tls={len(self._all_tls_ids)}, '
                    f'total_batches={total_batches}, batch_size={self.tls_batch_size}, '
                    f'keep_batch_sensors={self.keep_batch_sensors}, '
                    f'reuse_batch_sensors={self.reuse_batch_sensors}'
                )

    def _initialize_current_batch(self) -> List[str]:
        """????????????????????
        
        Returns:
            ???????????? tls_id ???
        """
        if self._tls_init_obs is None:
            return []
        
        start_idx = self._current_batch_index * self.tls_batch_size
        end_idx = min(start_idx + self.tls_batch_size, len(self._all_tls_ids))
        batch_tls_ids = self._all_tls_ids[start_idx:end_idx]
        
        for tls_id in batch_tls_ids:
            if tls_id not in self._initialized_tls_ids:
                self._initialize_tls_elements(tls_id, self._tls_init_obs[tls_id])
                self._initialized_tls_ids.add(tls_id)
        
        logger.info(f'SIM: ????????{self._current_batch_index + 1}????? {batch_tls_ids}')
        return batch_tls_ids
    
    def switch_to_batch(self, batch_index: int) -> List[str]:
        """?????????????????????????????????????
        
        Args:
            batch_index: ????????????0?????
            
        Returns:
            ?????? tls_id ???
        """
        if self.tls_batch_size is None:
            logger.warning('SIM: ?????????????????????')
            return list(self._initialized_tls_ids)
        
        total_batches = (len(self._all_tls_ids) + self.tls_batch_size - 1) // self.tls_batch_size
        if batch_index < 0 or batch_index >= total_batches:
            logger.warning(f'SIM: ?????? {batch_index} ?????? [0, {total_batches-1}]')
            return []
        
        if (
            (not getattr(self, 'keep_batch_sensors', True))
            and getattr(self, 'reuse_batch_sensors', False)
            and batch_index != self._current_batch_index
        ):
            self._current_batch_index = batch_index
            return self._reuse_tls_elements_for_current_batch()

        if (not getattr(self, 'keep_batch_sensors', True)) and batch_index != self._current_batch_index:
            self._destroy_current_batch()

        self._current_batch_index = batch_index
        return self._initialize_current_batch()
    
    def switch_to_next_batch(self) -> List[str]:
        """?????????????????
        
        Returns:
            ?????? tls_id ???
        """
        if self.tls_batch_size is None:
            return list(self._initialized_tls_ids)
        
        total_batches = (len(self._all_tls_ids) + self.tls_batch_size - 1) // self.tls_batch_size
        next_batch = (self._current_batch_index + 1) % total_batches
        return self.switch_to_batch(next_batch)
    
    def _destroy_current_batch(self) -> None:
        """Destroy sensors for the current batch."""
        start_idx = self._current_batch_index * self.tls_batch_size
        end_idx = min(start_idx + self.tls_batch_size, len(self._all_tls_ids))
        batch_tls_ids = self._all_tls_ids[start_idx:end_idx]
        
        for tls_id in batch_tls_ids:
            # ?????????????????????????
            elements_to_remove = [eid for eid in self._tls_elements.keys() if eid.startswith(f'{tls_id}_')]
            for element_id in elements_to_remove:
                self._tls_elements[element_id].remove_node()
                del self._tls_elements[element_id]
            self._initialized_tls_ids.discard(tls_id)
        
        logger.info(f'SIM: ???????{self._current_batch_index + 1} ??????')
    
    def get_current_batch_tls_ids(self) -> List[str]:
        """??????????????D???"""
        if self.tls_batch_size is None:
            return self._all_tls_ids
        
        start_idx = self._current_batch_index * self.tls_batch_size
        end_idx = min(start_idx + self.tls_batch_size, len(self._all_tls_ids))
        return self._all_tls_ids[start_idx:end_idx]
    
    def get_batch_info(self) -> Dict:
        """?????????"""
        if self.tls_batch_size is None:
            return {
                'mode': 'all',
                'total_tls': len(self._all_tls_ids),
                'initialized_tls': len(self._initialized_tls_ids)
            }
        
        total_batches = (len(self._all_tls_ids) + self.tls_batch_size - 1) // self.tls_batch_size
        return {
            'mode': 'batch',
            'batch_size': self.tls_batch_size,
            'current_batch': self._current_batch_index,
            'total_batches': total_batches,
            'total_tls': len(self._all_tls_ids),
            'initialized_tls': len(self._initialized_tls_ids),
            'keep_batch_sensors': getattr(self, 'keep_batch_sensors', True),
            'reuse_batch_sensors': getattr(self, 'reuse_batch_sensors', False),
            'current_batch_tls': self.get_current_batch_tls_ids()
        }

    def _current_batch_tls_elements(self) -> Dict:
        if self.tls_batch_size is None:
            return self._tls_elements
        current_tls_ids = set(self.get_current_batch_tls_ids())
        return {
            element_id: element
            for element_id, element in self._tls_elements.items()
            if element_id.rsplit('_', 1)[0] in current_tls_ids
        }

    def _initialize_tls_elements(self, tls_id, tls_info) -> None:
        """???????????????????
        """
        sensor_types = self.sensor_config['tls'][tls_id].get('sensor_types', []) # ?????????(tls_id)????????
        tls_camera_height = self.sensor_config['tls'][tls_id].get('tls_camera_height', 10) # ?????????
        # ??????????? ?????????????????
        sorted_road_ids = sorted(tls_info['in_roads_heading'], key=tls_info['in_roads_heading'].get)

        for index, road_id in enumerate(sorted_road_ids):
            tls_element_id = f'{tls_id}_{index}'
            position = calculate_center_point(tls_info['in_road_stop_line'][road_id])
            heading = tls_info['in_roads_heading'][road_id]
            element = TLS3DElement(
                fig_width = self.fig_width,
                fig_height = self.fig_height,
                fig_resolution = self.resolution,
                element_id = tls_element_id, 
                element_position = position, 
                element_heading = heading, 
                root_np = self.root_np, 
                showbase_instance = self.showbase_instance,
                tls_camera_height=tls_camera_height
            ) # ??????????????element
            element.attach_sensors_to_element(sensor_types)
            self._tls_elements[tls_element_id] = element      

    def _reuse_tls_elements_for_current_batch(self) -> List[str]:
        if self._tls_init_obs is None:
            return []

        pool = list(getattr(self, '_tls_element_pool', [])) + list(self._tls_elements.values())
        self._tls_elements = {}
        self._initialized_tls_ids = set()

        start_idx = self._current_batch_index * self.tls_batch_size
        end_idx = min(start_idx + self.tls_batch_size, len(self._all_tls_ids))
        batch_tls_ids = self._all_tls_ids[start_idx:end_idx]

        for tls_id in batch_tls_ids:
            tls_info = self._tls_init_obs[tls_id]
            sensor_types = self.sensor_config['tls'][tls_id].get('sensor_types', [])
            tls_camera_height = self.sensor_config['tls'][tls_id].get('tls_camera_height', 10)
            sorted_road_ids = sorted(tls_info['in_roads_heading'], key=tls_info['in_roads_heading'].get)

            for index, road_id in enumerate(sorted_road_ids):
                tls_element_id = f'{tls_id}_{index}'
                position = calculate_center_point(tls_info['in_road_stop_line'][road_id])
                heading = tls_info['in_roads_heading'][road_id]

                if pool:
                    element = pool.pop()
                    self._retarget_tls_element(
                        element=element,
                        tls_element_id=tls_element_id,
                        position=position,
                        heading=heading,
                        tls_camera_height=tls_camera_height,
                    )
                else:
                    element = TLS3DElement(
                        fig_width=self.fig_width,
                        fig_height=self.fig_height,
                        fig_resolution=self.resolution,
                        element_id=tls_element_id,
                        element_position=position,
                        element_heading=heading,
                        root_np=self.root_np,
                        showbase_instance=self.showbase_instance,
                        tls_camera_height=tls_camera_height,
                    )
                    element.attach_sensors_to_element(sensor_types)

                self._tls_elements[tls_element_id] = element
            self._initialized_tls_ids.add(tls_id)

        self._tls_element_pool = pool
        return batch_tls_ids

    def _retarget_tls_element(self, element, tls_element_id, position, heading, tls_camera_height) -> None:
        element.element_id = tls_element_id
        element.element_position = position
        element.element_heading = heading
        element.tls_camera_height = tls_camera_height

        for sensor in getattr(element, 'sensors', {}).values():
            if hasattr(sensor, 'init_actor'):
                sensor.init_actor(
                    element_pose=element.get_element_pose_from_center(),
                    height=tls_camera_height,
                )


    def _sync(self, tshub_obs):
        # ????????????
        veh_ids, aircraft_ids = self.update_elements(tshub_obs)
        
        # ????????vehicle ??aircraft
        self.remove_missing_elements(veh_ids, self._vehicle_elements, 'vehicle')
        self.remove_missing_elements(aircraft_ids, self._aircraft_elements, 'aircraft')
        
        # ??? camera
        logger.info(f'SIM: Update All Sensors Positions.')
        if self.step_task_manager:
            self.showbase_instance.taskMgr.step()
        
        # ???????????
        self.showbase_instance.graphicsEngine.renderFrame()

        # ??? camera ?????
        _sensors = {
            **self.collect_sensors(self._current_batch_tls_elements()), 
            **self.collect_sensors(self._vehicle_elements), 
            **self.collect_sensors(self._aircraft_elements)
        }
        return _sensors

    def update_elements(self, tshub_obs):
        """??? SUMO ????????vehicle ??aircraft ????? ????????
        1. self._manage_vehicle_element: ?????????
        2. self._manage_aircraft_element ???????????
        """
        # ???????????? vehicles ??aircrafts ??ids
        # ?????objects, ?????
        # - ?????????????object ??? (?????????????????????????????)
        # - ????????????, ????????panda3d ????????????
        veh_ids, aircraft_ids = set(), set()

        # 1. ???????????(????????SUMO ???????? tshub3d ????????
        for veh_id, veh_info in tshub_obs.get('vehicle', {}).items():
            veh_ids.add(veh_id)
            self._manage_vehicle_element(veh_id, veh_info)

        # 2. ???????????? (????????SUMO ???????? tshub3d ????????
        for aircraft_id, aircraft_info in tshub_obs.get('aircraft', {}).items():
            aircraft_ids.add(aircraft_id)
            self._manage_aircraft_element(aircraft_id, aircraft_info)         

        return veh_ids, aircraft_ids

    def _manage_vehicle_element(self, veh_id, veh_info) -> None:
        # ??????????????????
        element = self._vehicle_elements.get(veh_id)
        if not element: # ???????????
            element = Vehicle3DElement(
                fig_width = self.fig_width,
                fig_height = self.fig_height,
                fig_resolution = self.resolution,
                vehicle_model=self.vehicle_model,
                veh_id=veh_id, 
                veh_type=veh_info['vehicle_type'], 
                veh_pos=veh_info['position'], 
                veh_heading=veh_info['heading'], 
                veh_length=veh_info['length'], 
                root_np=self.root_np, 
                showbase_instance=self.showbase_instance
            )
            element.create_node()
            element.begin_rendering_node()
            # ???????????????????????
            if veh_id in self.sensor_config.get('vehicle', {}):
                sensor_types = self.sensor_config['vehicle'][veh_id].get('sensor_types', []) # ????????????????
                element.attach_sensors_to_element(sensor_types)
            self._vehicle_elements[veh_id] = element
        else:
            element.update_node(
                veh_position=veh_info['position'], 
                veh_heading=veh_info['heading'], 
                veh_type=veh_info['vehicle_type']
            )

    def _manage_aircraft_element(self, aircraft_id, aircraft_info) -> None:
        heading = math.degrees(vec_to_radians(vec_2d(aircraft_info['heading']))) % 360
        element = self._aircraft_elements.get(aircraft_id)
        if not element:
            element = Aircraft3DElement(
                fig_width = self.fig_width,
                fig_height = self.fig_height,
                fig_resolution = self.resolution,
                aircraft_id=aircraft_id, 
                aircraft_pos=aircraft_info['position'], 
                aircraft_heading=heading, 
                root_np=self.root_np, 
                showbase_instance=self.showbase_instance
            )
            if aircraft_id in self.sensor_config.get('aircraft', {}):
                sensor_types = self.sensor_config['aircraft'][aircraft_id].get('sensor_types', []) # ?????????????????
                element.attach_sensors_to_element(sensor_types)
            self._aircraft_elements[aircraft_id] = element
        else:
            element.update_sensor(aircraft_info['position'], heading)

    def remove_missing_elements(self, current_ids, elements, element_type):
        missing_ids = set(elements) - current_ids # ????????id
        for id in missing_ids:
            elements[id].remove_node() # tarffic element ?????remove node
            del elements[id]
            logger.info(f'SIM: 3D, Del {element_type} {id} since it leaves the scenario.')

    def collect_sensors(self, elements):
        # ???????????
        sensor_outputs = {} # ????????????
        for id, element in elements.items():
            _sensor_output = element.get_sensor()
            if _sensor_output:
                sensor_outputs[id] = _sensor_output
        return sensor_outputs

