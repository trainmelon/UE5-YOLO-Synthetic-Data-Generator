import unreal
import os
import random
import traceback
import math
import time

# ==========================================
# 僵尸杀手：确保每次运行都是全新的干净环境
# ==========================================
if hasattr(unreal, "_golden_batch_tick_handle") and unreal._golden_batch_tick_handle:
    try:
        unreal.unregister_slate_post_tick_callback(unreal._golden_batch_tick_handle)
        unreal.log_warning(">> 已清理后台旧任务。")
    except: pass
unreal._golden_batch_tick_handle = None

class ExhaustiveBirdseyeGenerator:
    def __init__(self):
        self.output_dir = "D:/Dataset_Output_Test/data_github"
        self.img_dir = os.path.join(self.output_dir, "images")
        self.lbl_dir = os.path.join(self.output_dir, "labels")
        self.kill_switch = os.path.join(self.output_dir, "STOP.txt")
        
        self.res_x = 1920.0
        self.res_y = 1080.0
        self.fov = 90.0
        
        self.class_map = {
            "Turbine": 0, "Tower": 1, "Car": 2, "Ship": 3
        }
        
        self.total_images = 30  # 最小单元测试
        self.current_count = 0
        
        self.wait_time = 3.0           
        self.save_time = 1.0           
        self.warmup_time_first = 8.0   
        self.warmup_time_light = 5.0   
        
        self.state = "COLD_START" 
        self.target_time = 0.0
        
        self.all_valid_actors = [] 
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
                    comp.set_world_rotation(unreal.Rotator(pitch=target_pitch, yaw=random.uniform(0, 360), roll=0.0), False, False)
        except: pass

    def project_3d_to_2d_math(self, target_loc, cam_loc, cam_rot):
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
            
        plane_dist = (self.res_x / 2.0) / math.tan(math.radians(self.fov / 2.0))
        proj_y = (local_y / local_x) * plane_dist
        proj_z = (local_z / local_x) * plane_dist
        
        screen_x = (self.res_x / 2.0) + proj_y
        screen_y = (self.res_y / 2.0) - proj_z
        
        return unreal.Vector2D(screen_x, screen_y)

    def check_occlusion_for_actor(self, actor):
        origin, extent = actor.get_actor_bounds(False, False)
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
                hit = unreal.SystemLibrary.sphere_trace_single(
                    world_context_object=self.world, start=self.cam_loc, end=pt,
                    radius=15.0, trace_channel=unreal.TraceTypeQuery.TRACE_TYPE_QUERY1,
                    trace_complex=True, actors_to_ignore=[actor], 
                    draw_debug_type=unreal.DrawDebugTrace.NONE, ignore_self=True
                )
                if not (hit and getattr(hit, 'blocking_hit', False) and hit.actor):
                    visible_count += 1
            except: pass 

        if (visible_count / 9.0) >= 0.3: return True 
        else: return False

    def calculate_yolo_label_for_actor(self, actor, class_id):
        origin, extent = actor.get_actor_bounds(False, False)
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

        if max_x < 0 or min_x > self.res_x or max_y < 0 or min_y > self.res_y: return None

        clamped_min_x = max(0, min_x)
        clamped_min_y = max(0, min_y)
        clamped_max_x = min(self.res_x, max_x)
        clamped_max_y = min(self.res_y, max_y)

        pixel_w = clamped_max_x - clamped_min_x
        pixel_h = clamped_max_y - clamped_min_y

        if pixel_w < 15 or pixel_h < 15: return None
        if not self.check_occlusion_for_actor(actor): return None

        center_x = ((clamped_min_x + clamped_max_x) / 2.0) / self.res_x
        center_y = ((clamped_min_y + clamped_max_y) / 2.0) / self.res_y
        width = pixel_w / self.res_x
        height = pixel_h / self.res_y

        return f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n"

    def get_all_tags(self, actor):
        tags = [str(tag) for tag in actor.tags]
        try:
            for comp in actor.get_components_by_class(unreal.ActorComponent.static_class()): 
                tags.extend([str(tag) for tag in comp.component_tags])
        except: pass
        return list(set(tags))

    def _on_tick(self, delta_time):
        try:
            if os.path.exists(self.kill_switch):
                self.stop()
                return

            if self.current_count >= self.total_images:
                unreal.log("=== 最小单元测试完毕 ===")
                self.stop()
                return

            now = time.time()

            if self.state == "COLD_START":
                unreal.log("执行引擎热身...")
                self.target_time = now + self.warmup_time_first
                self.state = "WAIT_COLD_START" 

            elif self.state == "WAIT_COLD_START":
                if now >= self.target_time: self.state = "MOVE_CAMERA" 

            elif self.state == "MOVE_CAMERA":
                current_phase_wait = self.wait_time
                if self.current_count == 0: self.force_change_lighting(0)
                
                all_actors = unreal.GameplayStatics.get_all_actors_of_class(self.world, unreal.Actor.static_class())
                
                # 【核心修复 1：分层抽样机制】将所有地物按类别收纳进不同抽屉，解决车辆霸屏问题
                categorized_actors = {0: [], 1: [], 2: [], 3: []}
                self.all_valid_actors = [] # 用于后期的全局穷举标注
                
                for actor in all_actors:
                    for tag in self.get_all_tags(actor):
                        if tag in self.class_map:
                            cid = self.class_map[tag]
                            categorized_actors[cid].append(actor)
                            self.all_valid_actors.append((actor, cid))
                            break
                
                if not self.all_valid_actors: return self.stop()

                # 先随机挑一个“类别”，再从这个类别里随机挑一个“物体”。保证风机和车被拍到的概率永远是 1:1
                available_classes = [cid for cid, actors in categorized_actors.items() if len(actors) > 0]
                anchor_class_id = random.choice(available_classes)
                anchor_actor = random.choice(categorized_actors[anchor_class_id])
                
                origin, extent = anchor_actor.get_actor_bounds(False, False)
                max_size = max(extent.x, extent.y, extent.z)
                
# 【核心修复 2：类别自适应焦距】对不同体型地物实施精准的独立运镜
                if anchor_class_id in [0, 1]:  
                    # Turbine(风机) & Tower(铁塔) -> 细长高耸型：根据高度(Z)拉近
                    distance = random.uniform(extent.z * 2.0, extent.z * 5.0)
                    
                elif anchor_class_id == 2:  
                    # Car(汽车) -> 小型扁平物：由于尺寸小，需要较高倍数（4~10倍）来拍出航拍车流感
                    distance = random.uniform(max_size * 4.0, max_size * 10.0)
                    
                elif anchor_class_id == 3:  
                    # Ship(轮船) -> 巨型扁平物：本身极长(X极大)，如果乘10倍会飞到平流层。必须死死压住倍数！
                    # 仅拉远 1.5 到 3.5 倍，确保大船能填满大半个屏幕，小船也能看得清
                    distance = random.uniform(max_size * 1.5, max_size * 3.5)
                
                pitch = random.uniform(-89.0, -60.0)
                yaw = random.uniform(0.0, 360.0)
                
                pitch_rad = math.radians(pitch)
                yaw_rad = math.radians(yaw)
                dir_x = math.cos(pitch_rad) * math.cos(yaw_rad)
                dir_y = math.cos(pitch_rad) * math.sin(yaw_rad)
                dir_z = math.sin(pitch_rad)
                
                offset_x = random.uniform(-extent.x, extent.x)
                offset_y = random.uniform(-extent.y, extent.y)
                target_focus = origin + unreal.Vector(offset_x, offset_y, 0)
                
                # 为高耸物体增加一个Z轴安全偏移，防止摄像机由于靠得太近，一头扎进铁塔顶部模型里
                if anchor_class_id in [0, 1]:
                    target_focus.z += extent.z * 0.5 
                
                self.cam_loc = target_focus - unreal.Vector(dir_x, dir_y, dir_z) * distance
                self.cam_rot = unreal.Rotator(pitch=pitch, yaw=yaw, roll=0.0)
                
                self.editor_subsystem.set_level_viewport_camera_info(self.cam_loc, self.cam_rot)
                
                self.target_time = now + current_phase_wait
                self.state = "WAIT_LOADING"

            elif self.state == "WAIT_LOADING":
                if now >= self.target_time:
                    final_label_str = ""
                    visible_count = 0
                    
                    for actor, class_id in self.all_valid_actors:
                        single_label = self.calculate_yolo_label_for_actor(actor, class_id)
                        if single_label:
                            final_label_str += single_label
                            visible_count += 1
                    
                    if final_label_str != "":
                        file_basename = f"frame_test_{self.current_count:04d}"
                        img_path = os.path.join(self.img_dir, file_basename + ".png")
                        lbl_path = os.path.join(self.lbl_dir, file_basename + ".txt")

                        with open(lbl_path, "w") as f: f.write(final_label_str)

                        unreal.AutomationLibrary.take_high_res_screenshot(int(self.res_x), int(self.res_y), img_path)
                        unreal.log(f"-> 抓拍成功！本张图框出 {visible_count} 个目标。保存: {file_basename}")
                        
                        self.target_time = now + self.save_time
                        self.state = "WAIT_SCREENSHOT"
                    else:
                        self.state = "MOVE_CAMERA"
            
            elif self.state == "WAIT_SCREENSHOT":
                if now >= self.target_time:
                    self.current_count += 1
                    self.state = "MOVE_CAMERA"

        except Exception as e:
            unreal.log_error(f"发生错误: {str(e)}\n{traceback.format_exc()}")
            self.stop()

    def stop(self):
        if unreal._golden_batch_tick_handle:
            unreal.unregister_slate_post_tick_callback(unreal._golden_batch_tick_handle)
            unreal._golden_batch_tick_handle = None

    def start(self):
        self.ensure_directories()
        unreal.log(f"=== 开始 穷举俯视版 MVP 单元测试 (v2.1 分层抽样版) ===")
        unreal._golden_batch_tick_handle = unreal.register_slate_post_tick_callback(self._on_tick)

ExhaustiveBirdseyeGenerator().start()