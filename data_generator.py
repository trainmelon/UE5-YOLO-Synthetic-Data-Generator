import unreal
import os
import random
import traceback
import math
import time

# ==========================================
# 后台可能的残留实例清理，确保本次运行环境干净
# ==========================================
old_handles = [
    "_global_batch_tick_handle", "_active_pie_generator", 
    "_standard_generator_tick", "_golden_batch_tick_handle", 
    "_heavy_armor_tick", "_final_golden_tick", "_golden_production_tick",
    "_ultimate_golden_production"
]
for attr in old_handles:
    if hasattr(unreal, attr):
        handle = getattr(unreal, attr)
        if handle:
            try: unreal.unregister_slate_post_tick_callback(handle)
            except: pass
        setattr(unreal, attr, None)

class UltimateGoldenProduction:
    def __init__(self):
        # 1. 修改为正式数据集输出路径
        self.output_dir = "D:/Dataset_Output_Full/data"
        self.img_dir = os.path.join(self.output_dir, "images")
        self.lbl_dir = os.path.join(self.output_dir, "labels")
        self.kill_switch = os.path.join(self.output_dir, "STOP.txt")
        
        self.res_x = 1920.0
        self.res_y = 1080.0
        self.fov = 90.0
        
        self.class_map = {
            "Turbine": 0, "Tower": 1, "Car": 2, "Ship": 3
        }
        
        # 2. 生成总量调整（此处默认为 5000 张）
        self.total_images = 5000  
        self.current_count = 0
        
        # 时序控制，用于兼容低性能机器，确保每一步都有足够时间完成，可以适当调整参数
        self.wait_time = 3.0           
        self.save_time = 1.0           
        self.warmup_time_first = 8.0   
        self.warmup_time_light = 5.0   
        
        self.target_time = 0.0
        self.state = "COLD_START" 
        
        self.current_actor = None
        self.current_class_id = None
        self.cam_loc = None
        self.cam_rot = None
        
        self.editor_subsystem = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
        self.world = self.editor_subsystem.get_editor_world()

    def ensure_directories(self):
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.lbl_dir, exist_ok=True)
        if os.path.exists(self.kill_switch):
            os.remove(self.kill_switch)

    def force_change_lighting(self, phase):
        pitch_map = {0: -15.0, 1: -80.0, 2: -5.0}
        target_pitch = pitch_map.get(phase, -45.0)
        time_map = {0: 7.0, 1: 12.0, 2: 17.5}
        
        try:
            actors = unreal.GameplayStatics.get_all_actors_of_class(self.world, unreal.Actor.static_class())
            for actor in actors:
                name = actor.get_name()
                if "SunSky" in name or "CesiumSunSky" in name:
                    actor.set_editor_property("SolarTime", time_map[phase])
                    try: actor.call_function_by_name("UpdateSun")
                    except: pass
        except: pass

        try:
            for actor in actors:
                light_comps = actor.get_components_by_class(unreal.DirectionalLightComponent.static_class())
                for comp in light_comps:
                    new_rot = unreal.Rotator(pitch=target_pitch, yaw=random.uniform(0, 360), roll=0.0)
                    comp.set_world_rotation(new_rot, False, False)
        except: pass
            
        try:
            sky_lights = unreal.GameplayStatics.get_all_actors_of_class(self.world, unreal.SkyLight.static_class())
            for sky in sky_lights:
                sky_comp = sky.get_component_by_class(unreal.SkyLightComponent.static_class())
                if sky_comp: sky_comp.recapture_sky()
        except: pass

    def project_3d_to_2d_math(self, target_loc, cam_loc, cam_rot):
        #  3D-2D 投影算法
        dx = target_loc.x - cam_loc.x
        dy = target_loc.y - cam_loc.y
        dz = target_loc.z - cam_loc.z
        
        yaw_rad = math.radians(cam_rot.yaw)
        pitch_rad = math.radians(cam_rot.pitch)
        
        cy = math.cos(yaw_rad)
        sy = math.sin(yaw_rad)
        cp = math.cos(pitch_rad)
        sp = math.sin(pitch_rad)
        
        fx, fy, fz = cy * cp, sy * cp, sp
        rx, ry, rz = -sy, cy, 0.0
        ux, uy, uz = -cy * sp, -sy * sp, cp
        
        local_x = dx * fx + dy * fy + dz * fz
        local_y = dx * rx + dy * ry + dz * rz
        local_z = dx * ux + dy * uy + dz * uz
        
        if local_x <= 0: return None
            
        half_fov_rad = math.radians(self.fov / 2.0)
        plane_dist = (self.res_x / 2.0) / math.tan(half_fov_rad)
        
        proj_y = (local_y / local_x) * plane_dist
        proj_z = (local_z / local_x) * plane_dist
        
        screen_x = (self.res_x / 2.0) + proj_y
        screen_y = (self.res_y / 2.0) - proj_z
        
        return unreal.Vector2D(screen_x, screen_y)

    def calculate_yolo_label(self):
        origin, extent = self.current_actor.get_actor_bounds(False, False)
        corners = [
            origin + unreal.Vector(extent.x, extent.y, extent.z),
            origin + unreal.Vector(extent.x, extent.y, -extent.z),
            origin + unreal.Vector(extent.x, -extent.y, extent.z),
            origin + unreal.Vector(extent.x, -extent.y, -extent.z),
            origin + unreal.Vector(-extent.x, extent.y, extent.z),
            origin + unreal.Vector(-extent.x, extent.y, -extent.z),
            origin + unreal.Vector(-extent.x, -extent.y, extent.z),
            origin + unreal.Vector(-extent.x, -extent.y, -extent.z)
        ]

        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        
        valid_points = 0
        for corner in corners:
            screen_pos = self.project_3d_to_2d_math(corner, self.cam_loc, self.cam_rot)
            if screen_pos:
                valid_points += 1
                min_x = min(min_x, screen_pos.x)
                min_y = min(min_y, screen_pos.y)
                max_x = max(max_x, screen_pos.x)
                max_y = max(max_y, screen_pos.y)
        
        if valid_points == 0: return None

        min_x = max(0, min_x); min_y = max(0, min_y)
        max_x = min(self.res_x, max_x); max_y = min(self.res_y, max_y)

        if min_x >= max_x or min_y >= max_y: return None

        center_x = ((min_x + max_x) / 2.0) / self.res_x
        center_y = ((min_y + max_y) / 2.0) / self.res_y
        width = (max_x - min_x) / self.res_x
        height = (max_y - min_y) / self.res_y

        return f"{self.current_class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n"

    def get_all_tags(self, actor):
        tags = [str(tag) for tag in actor.tags]
        try:
            components = actor.get_components_by_class(unreal.ActorComponent.static_class())
            for comp in components: tags.extend([str(tag) for tag in comp.component_tags])
        except: pass
        return list(set(tags))

    def check_occlusion_multi_ray(self):
        #  9点球体射线遮挡检测
        origin, extent = self.current_actor.get_actor_bounds(False, False)
        test_points = [
            origin,
            origin + unreal.Vector(extent.x, extent.y, extent.z),
            origin + unreal.Vector(extent.x, extent.y, -extent.z),
            origin + unreal.Vector(extent.x, -extent.y, extent.z),
            origin + unreal.Vector(extent.x, -extent.y, -extent.z),
            origin + unreal.Vector(-extent.x, extent.y, extent.z),
            origin + unreal.Vector(-extent.x, extent.y, -extent.z),
            origin + unreal.Vector(-extent.x, -extent.y, extent.z),
            origin + unreal.Vector(-extent.x, -extent.y, -extent.z)
        ]
        
        visible_count = 0
        for pt in test_points:
            try:
                hit_result = unreal.SystemLibrary.sphere_trace_single(
                    world_context_object=self.world, start=self.cam_loc, end=pt,
                    radius=15.0, trace_channel=unreal.TraceTypeQuery.TRACE_TYPE_QUERY1,
                    trace_complex=True, actors_to_ignore=[self.current_actor], 
                    draw_debug_type=unreal.DrawDebugTrace.NONE, ignore_self=True
                )
                if not (hit_result and getattr(hit_result, 'blocking_hit', False) and hit_result.actor):
                    visible_count += 1
            except: pass 

        visibility_ratio = visible_count / 9.0
        if visibility_ratio >= 0.4: return False 
        else: return True 

    def _on_tick(self, delta_time):
        try:
            if os.path.exists(self.kill_switch):
                unreal.log_error("检测到 STOP.txt，程序已安全终止！")
                self.stop()
                return

            if self.current_count >= self.total_images:
                unreal.log(f"=== 成功完成 {self.total_images} 张完整数据集生成！ ===")
                self.stop()
                return

            now = time.time()

            if self.state == "COLD_START":
                unreal.log(f"执行初始引擎热身与底图加载，请等待 {self.warmup_time_first} 秒...")
                self.target_time = now + self.warmup_time_first
                self.state = "WAIT_COLD_START" 
                return

            elif self.state == "WAIT_COLD_START":
                if now >= self.target_time: self.state = "MOVE_CAMERA" 
                return

            elif self.state == "MOVE_CAMERA":
                current_phase_wait = self.wait_time
                
                # 光照阶梯变化
                if self.current_count == 0:
                    unreal.log(">> 切换至【早晨】光照阶段")
                    self.force_change_lighting(0)
                    current_phase_wait = self.warmup_time_light
                elif self.current_count == self.total_images // 3:
                    unreal.log(">> 切换至【正午】光照阶段")
                    self.force_change_lighting(1)
                    current_phase_wait = self.warmup_time_light
                elif self.current_count == (self.total_images * 2) // 3:
                    unreal.log(">> 切换至【黄昏】光照阶段")
                    self.force_change_lighting(2)
                    current_phase_wait = self.warmup_time_light

                all_actors = unreal.GameplayStatics.get_all_actors_of_class(self.world, unreal.Actor.static_class())
                categorized_targets = {0:[], 1:[], 2:[], 3:[]}
                for actor in all_actors:
                    tags = self.get_all_tags(actor)
                    for tag in tags:
                        if tag in self.class_map:
                            class_id = self.class_map[tag]
                            categorized_targets[class_id].append(actor)
                            break
                
                available_classes = [cid for cid, actors in categorized_targets.items() if len(actors) > 0]
                if not available_classes:
                    self.stop()
                    return

                self.current_class_id = random.choice(available_classes)
                self.current_actor = random.choice(categorized_targets[self.current_class_id])
                class_name = list(self.class_map.keys())[list(self.class_map.values()).index(self.current_class_id)]

                unreal.log(f"[{self.current_count + 1}/{self.total_images}] 锁定目标: {class_name}")

                origin, extent = self.current_actor.get_actor_bounds(False, False)
                
                if class_name in ["Tower", "Turbine"]:
                    radius = random.uniform(extent.z * 3.0, extent.z * 6.0)
                    height_offset = random.uniform(0.0, extent.z * 1.5)
                else:
                    max_size = max(extent.x, extent.y, extent.z) 
                    radius = random.uniform(max_size * 2.0, max_size * 5.0)
                    height_offset = random.uniform(max_size * 0.2, max_size * 2.0)
                
                angle = random.uniform(0, 360)
                cam_x = origin.x + radius * math.cos(math.radians(angle))
                cam_y = origin.y + radius * math.sin(math.radians(angle))
                
                min_ground_z = origin.z - extent.z * 0.8
                cam_z = max(min_ground_z, origin.z + height_offset)
                self.cam_loc = unreal.Vector(cam_x, cam_y, cam_z)
                
                dx = origin.x - self.cam_loc.x
                dy = origin.y - self.cam_loc.y
                dz = origin.z - self.cam_loc.z
                
                yaw = math.degrees(math.atan2(dy, dx))
                pitch = math.degrees(math.atan2(dz, math.sqrt(dx**2 + dy**2)))
                self.cam_rot = unreal.Rotator(pitch=pitch, yaw=yaw, roll=0.0)
                
                self.editor_subsystem.set_level_viewport_camera_info(self.cam_loc, self.cam_rot)

                self.target_time = now + current_phase_wait
                self.state = "WAIT_LOADING"

            elif self.state == "WAIT_LOADING":
                if now >= self.target_time:
                    if self.check_occlusion_multi_ray():
                        self.state = "MOVE_CAMERA"
                        return

                    label_str = self.calculate_yolo_label()
                    
                    if label_str:
                        # 3. 规范化文件命名：如 frame_001234.png
                        file_basename = f"frame_{self.current_count:06d}"
                        img_path = os.path.join(self.img_dir, file_basename + ".png")
                        lbl_path = os.path.join(self.lbl_dir, file_basename + ".txt")

                        with open(lbl_path, "w") as f: f.write(label_str)

                        unreal.AutomationLibrary.take_high_res_screenshot(int(self.res_x), int(self.res_y), img_path)
                        unreal.log(f"-> 快门按下，保存: {file_basename}")
                        
                        self.target_time = now + self.save_time
                        self.state = "WAIT_SCREENSHOT"
                        
                    else:
                        self.state = "MOVE_CAMERA"
            
            elif self.state == "WAIT_SCREENSHOT":
                if now >= self.target_time:
                    # 4. 强制垃圾回收，降低内存压力
                    unreal.SystemLibrary.collect_garbage()
                    self.current_count += 1
                    self.state = "MOVE_CAMERA"

        except Exception as e:
            unreal.log_error(f"发生错误: {str(e)}\n{traceback.format_exc()}")
            self.stop()

    def stop(self):
        if unreal._ultimate_golden_production:
            unreal.unregister_slate_post_tick_callback(unreal._ultimate_golden_production)
            unreal._ultimate_golden_production = None

    def start(self):
        self.ensure_directories()
        unreal.log(f"=== 开始 完整数据集生成测试 (共 {self.total_images} 张) ===")
        unreal._ultimate_golden_production = unreal.register_slate_post_tick_callback(self._on_tick)

unreal._ultimate_golden_production = None
UltimateGoldenProduction().start()