import unreal
import random
import traceback
import math

class SplineCarSpawner:
    def __init__(self):
        self.base_mesh_path = "/Game/model/car/SM_TYPE{}_CAR"
        self.loaded_meshes = []
        self.spline_actor_name = "BP_RoadSpline"
        self.spacing = 1500.0  
        self.random_offset_y = 300.0  
        # 加大探测范围，防止样条线画得太高导致够不到地面
        self.trace_z_offset_up = 5000.0   
        self.trace_z_offset_down = 50000.0 

    def get_editor_world_ue5(self):
        editor_subsystem = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
        return editor_subsystem.get_editor_world()

    def load_car_meshes(self):
        try:
            for i in range(1, 10):
                asset_path = self.base_mesh_path.format(i)
                mesh = unreal.EditorAssetLibrary.load_asset(asset_path)
                if mesh and isinstance(mesh, unreal.StaticMesh):
                    self.loaded_meshes.append(mesh)
                else:
                    unreal.log_warning(f"跳过无效资产或路径: {asset_path}")
            
            if not self.loaded_meshes:
                unreal.log_error("未加载到任何汽车网格体！请检查路径。")
                return False
            return True
        except Exception as e:
            unreal.log_error(f"加载模型时发生严重 API 报错:\n{e}\n{traceback.format_exc()}")
            return False

    def get_spline_actors(self):
        try:
            world = self.get_editor_world_ue5()
            actors = unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor.static_class())
            spline_actors = []
            
            for actor in actors:
                if self.spline_actor_name in actor.get_name():
                    spline_actors.append(actor)
            return spline_actors
        except Exception as e:
            unreal.log_error(f"寻找样条线 Actor 时发生 API 报错:\n{e}")
            return []

    def spawn_along_spline(self):
        if not self.load_car_meshes():
            return

        spline_actors = self.get_spline_actors()
        if not spline_actors:
            unreal.log_error(f"场景中未找到名字包含 {self.spline_actor_name} 的 Actor。")
            return

        world = self.get_editor_world_ue5()
        total_spawned = 0

        for spline_actor in spline_actors:
            try:
                spline_comp = spline_actor.get_component_by_class(unreal.SplineComponent)
                if not spline_comp:
                    unreal.log_warning(f"{spline_actor.get_name()} 身上没有 Spline 组件，跳过。")
                    continue

                spline_length = spline_comp.get_spline_length()
                current_distance = 0.0

                unreal.log(f"正在处理 {spline_actor.get_name()}，总长度: {spline_length:.2f}")

                while current_distance < spline_length:
                    transform = spline_comp.get_transform_at_distance_along_spline(
                        current_distance, unreal.SplineCoordinateSpace.WORLD)
                    
                    base_loc = transform.translation
                    base_rot = transform.rotation.rotator()

                    yaw_rad = math.radians(base_rot.yaw)
                    right_vec_x = math.sin(yaw_rad)
                    right_vec_y = -math.cos(yaw_rad)
                    
                    offset_dist = random.uniform(-self.random_offset_y, self.random_offset_y)
                    target_x = base_loc.x + right_vec_x * offset_dist
                    target_y = base_loc.y + right_vec_y * offset_dist

                    trace_start = unreal.Vector(target_x, target_y, base_loc.z + self.trace_z_offset_up)
                    trace_end = unreal.Vector(target_x, target_y, base_loc.z - self.trace_z_offset_down)

                    hit_result = unreal.SystemLibrary.line_trace_single(
                        world_context_object=world,
                        start=trace_start,
                        end=trace_end,
                        trace_channel=unreal.TraceTypeQuery.TRACE_TYPE_QUERY1,
                        trace_complex=True,
                        actors_to_ignore=[spline_actor],
                        # 开启 Debug 射线，视口中保留 5 秒
                        draw_debug_type=unreal.DrawDebugTrace.FOR_DURATION,
                        ignore_self=True
                    )

                    # 修复：安全判断 hit_result 是否为 None
                    if hit_result and hit_result.b_blocking_hit:
                        ground_loc = hit_result.impact_point
                        selected_mesh = random.choice(self.loaded_meshes)
                        
                        if random.choice([True, False]):
                            final_yaw = base_rot.yaw
                        else:
                            final_yaw = base_rot.yaw + 180.0
                            
                        spawn_rot = unreal.Rotator(pitch=0.0, yaw=final_yaw, roll=0.0)

                        spawned_actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
                            unreal.StaticMeshActor.static_class(),
                            ground_loc,
                            spawn_rot
                        )

                        if spawned_actor:
                            spawned_actor.static_mesh_component.set_static_mesh(selected_mesh)
                            spawned_actor.tags = ["Car"]
                            total_spawned += 1

                    current_distance += self.spacing * random.uniform(0.8, 1.2)

            except Exception as e:
                unreal.log_error(f"处理样条线 {spline_actor.get_name()} 时发生 API 报错: {e}")
                unreal.log_error(traceback.format_exc())
                continue

        unreal.log(f"=== 沿样条线生成完毕，共成功生成 {total_spawned} 辆带 Car 标签的汽车 ===")

spawner = SplineCarSpawner()
spawner.spawn_along_spline()