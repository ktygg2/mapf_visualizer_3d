import bpy
import csv
import random
import mathutils
import math

# 기존 오브젝트 삭제
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# CSV 파일 경로 지정
csv_file_path = "/home/kty/ros2_ws/src/mapf_visualizer_3d/mapf_visualizer_3d/path_smoothing_simple.csv"

# CSV 데이터 불러오기
trajectories = {}
with open(csv_file_path, 'r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        agent_id = int(float(row['agent']))
        x, y, z = float(row['x']), float(row['y']), float(row['z'])
        if agent_id not in trajectories:
            trajectories[agent_id] = []
        trajectories[agent_id].append((x, y, z))

# 드론 궤적 및 애니메이션 생성
for agent_id, points in trajectories.items():
    # Curve 생성 및 위치 보정
    curve_data = bpy.data.curves.new(f"curve_{agent_id}", type='CURVE')
    curve_data.dimensions = '3D'
    spline = curve_data.splines.new('POLY')
    spline.points.add(len(points)-1)
    
    for i, (x, y, z) in enumerate(points):
        spline.points[i].co = (x, y, z, 1.0)
    
    curve_obj = bpy.data.objects.new(f"Trajectory_{agent_id}", curve_data)
    bpy.context.collection.objects.link(curve_obj)
    
    # 커브 원점을 첫 번째 포인트로 강제 이동
    curve_obj.location = points[0]
    offset_vector = mathutils.Vector(points[0]) * -1  # 벡터 변환 및 부호 반전
    curve_obj.data.transform(mathutils.Matrix.Translation(offset_vector))
    
    # 애니메이션 설정 (기존 코드 교체)
    curve_data.use_path = True  # Path Animation 활성화 (핵심!)
    curve_data.path_duration = 250
    curve_data.eval_time = 0
    curve_data.keyframe_insert(data_path="eval_time", frame=1)
    curve_data.eval_time = 250
    curve_data.keyframe_insert(data_path="eval_time", frame=250)

    # 드론 생성을 원점에서 시작
    bpy.ops.mesh.primitive_cube_add(size=0.275, location=(0,0,0))
    drone = bpy.context.object
    drone.name = f"Drone_{agent_id}"

    # 팔 큐브 생성 (원점 기준 오프셋)
    arm_objs = []
    for offset in [(-0.3, 0, 0), (0.3, 0, 0), (0, -0.3, 0), (0, 0.3, 0)]:
        bpy.ops.mesh.primitive_cube_add(size=0.1, location=offset)
        arm_objs.append(bpy.context.object)

    # 본체와 팔 모두 선택해서 합치기
    bpy.ops.object.select_all(action='DESELECT')
    drone.select_set(True)
    for arm in arm_objs:
        arm.select_set(True)
    bpy.context.view_layer.objects.active = drone
    bpy.ops.object.join()

    # Follow Path 제약 적용 (드론 전체)
    constraint = drone.constraints.new('FOLLOW_PATH')
    constraint.target = curve_obj
    constraint.use_curve_follow = True
    constraint.forward_axis = 'FORWARD_Y'
    constraint.up_axis = 'UP_Z'
    
    # 코드 마지막에 추가
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 250
    bpy.context.scene.frame_set(1)  # 첫 번째 프레임으로 이동
    
    # 충돌 거리 임계값(단위: m)
    collision_threshold = 0.3

    # 드론 오브젝트 리스트 (이름 패턴에 맞게 수정)
    drones = [obj for obj in bpy.context.scene.objects if obj.name.startswith("Drone_")]

    frame_start = bpy.context.scene.frame_start
    frame_end = bpy.context.scene.frame_end

    for frame in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(frame)
        for i in range(len(drones)):
            for j in range(i + 1, len(drones)):
                pos1 = drones[i].matrix_world.translation
                pos2 = drones[j].matrix_world.translation
                distance = (pos1 - pos2).length
                if distance < collision_threshold:
                    print(f"Frame {frame}: {drones[i].name}와 {drones[j].name} 충돌! (거리: {distance:.3f}m)")