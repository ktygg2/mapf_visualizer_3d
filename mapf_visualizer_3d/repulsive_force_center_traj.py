import pandas as pd
import numpy as np
import time
import os

# CSV 파일 경로
csv_path = '/home/kty/ros2_ws/src/mapf_visualizer_3d/scripts/result_interpolator.csv'

# CSV 파일 읽기
interpolated_points = pd.read_csv(csv_path)
interpolated_points['agent'] = interpolated_points['agent'].astype(int)

print("데이터 정보:")
print(f"에이전트 수: {interpolated_points['agent'].nunique()}")
print(f"시간 포인트 수: {interpolated_points['time'].nunique()}")
print(f"시간 범위: {interpolated_points['time'].min()} ~ {interpolated_points['time'].max()}")
print(interpolated_points.head())

# 파라미터 설정
R = 0.5
attractive_strength = 1.0
straight_line_strength = 0.3
n_replanning = 10
alpha = 0.3

time_steps = np.sort(interpolated_points['time'].unique())
agents = np.sort(interpolated_points['agent'].unique())

# 각 에이전트의 경로를 NumPy 배열로 저장 (agent_dict[agent] = (N, 4) 배열)
agent_dict = {}
agent_start_goals = {}
for agent in agents:
    agent_data = interpolated_points[interpolated_points['agent'] == agent].sort_values('time')
    arr = agent_data[['time', 'x', 'y', 'z']].values
    agent_dict[agent] = arr
    agent_start_goals[agent] = {
        'start': arr[0, 1:],
        'goal': arr[-1, 1:]
    }
    print(f"Agent {agent}: {arr[0, 1:]} → {arr[-1, 1:]}")

# 시간-에이전트 인덱스 매핑 배열 생성
time_idx_map = {t: i for i, t in enumerate(time_steps)}
agent_idx_map = {a: i for i, a in enumerate(agents)}
n_time = len(time_steps)
n_agent = len(agents)

# 전체 위치 배열 (time, agent, xyz)
positions = np.zeros((n_time, n_agent, 3))
for a_idx, agent in enumerate(agents):
    arr = agent_dict[agent]
    for row in arr:
        t_idx = time_idx_map[row[0]]
        positions[t_idx, a_idx, :] = row[1:]

# 속도 벡터 사전 (agent, time 인덱스 기반)
def calc_velocities(positions):
    velocities = np.zeros_like(positions)
    velocities[1:-1] = (positions[2:] - positions[:-2]) / (time_steps[2:, None] - time_steps[:-2, None])[:,:,None]
    velocities[0] = (positions[1] - positions[0]) / max(1, time_steps[1] - time_steps[0])
    velocities[-1] = (positions[-1] - positions[-2]) / max(1, time_steps[-1] - time_steps[-2])
    return velocities

# 인접점 중심 계산 (벡터화)
def get_neighbor_center_targets(pos_arr):
    prev = np.vstack([pos_arr[0], pos_arr[:-1]])
    next = np.vstack([pos_arr[1:], pos_arr[-1]])
    return (prev + next) / 2

# 직선 타겟 계산 (벡터화)
def get_straight_line_targets(start, goal, pos_arr):
    line_vec = goal - start
    line_len2 = np.dot(line_vec, line_vec)
    if line_len2 == 0:
        return np.tile(start, (len(pos_arr), 1))
    current_vecs = pos_arr - start
    t = np.clip(np.dot(current_vecs, line_vec) / line_len2, 0.0, 1.0)
    return start + np.outer(t, line_vec)

# 충돌 위험도 계산 (벡터화)
def calculate_collision_risk_vec(current_pos, other_positions, R):
    if len(other_positions) == 0:
        return 0.0
    dists = np.linalg.norm(other_positions - current_pos, axis=1)
    min_dist = np.min(dists)
    risk_threshold = 3 * R
    if min_dist >= risk_threshold:
        return 0.0
    else:
        return 1.0 - (min_dist / risk_threshold)

# 척력 계산 (벡터화)
def calculate_repulsive_force_vec(current_pos, other_positions, current_vel, other_vels, R=0.5):
    if len(other_positions) == 0:
        return np.zeros(3)
    rel_pos = other_positions - current_pos
    dists = np.linalg.norm(rel_pos, axis=1)
    min_distance = 2 * R
    rel_vel = other_vels - current_vel
    closing_speed = np.array([
        -np.dot(rv, rp)/d if d != 0 else 0
        for rv, rp, d in zip(rel_vel, rel_pos, dists)
    ])
    ttc = np.where(closing_speed > 0, dists / closing_speed, np.inf)
    rep_force = np.zeros((len(other_positions), 3))
    for i, d in enumerate(dists):
        if d == 0:
            rep_force[i] = np.array([1.0, 0.0, 0.0]) * (min_distance / 2)
        elif d >= min_distance:
            rep_force[i] = 0
        else:
            diff = min_distance - d
            force_mag = diff / 2
            if ttc[i] < 2.0:
                mv_dir = current_vel / np.linalg.norm(current_vel) if np.linalg.norm(current_vel) != 0 else np.zeros(3)
                rep_dir = -mv_dir
                rep_force[i] = rep_dir * force_mag
            else:
                unit_vec = -rel_pos[i] / d
                rep_force[i] = unit_vec * force_mag
    return np.sum(rep_force, axis=0)

# 인력 계산 (벡터화)
def calculate_combined_attractive_force_vec(current_pos, neighbor_center, straight_line_target, neighbor_strength, straight_strength, collision_risk):
    neighbor_disp = neighbor_center - current_pos
    neighbor_dist = np.linalg.norm(neighbor_disp)
    neighbor_force = (neighbor_disp / neighbor_dist) * (neighbor_strength * neighbor_dist) if neighbor_dist > 0 else np.zeros(3)
    straight_disp = straight_line_target - current_pos
    straight_dist = np.linalg.norm(straight_disp)
    straight_force = (straight_disp / straight_dist) * (straight_strength * straight_dist) if straight_dist > 0 else np.zeros(3)
    neighbor_weight = 0.7 + 0.3 * collision_risk
    straight_weight = 1.0 - 0.5 * collision_risk
    return neighbor_force * neighbor_weight + straight_force * straight_weight

# === 반복 전체 결과 누적 저장 ===
all_iterations_paths = []

start_time = time.time()
for iteration in range(n_replanning):
    print(f"\n=== Replanning Iteration {iteration+1} / {n_replanning} ===")
    modified_paths = []
    current_straight_strength = straight_line_strength * (1.0 + iteration * 0.2)
    velocities = calc_velocities(positions)
    for t_idx, t in enumerate(time_steps):
        current_positions = positions[t_idx]
        current_vels = velocities[t_idx]
        for a_idx, agent in enumerate(agents):
            current_pos = current_positions[a_idx]
            current_vel = current_vels[a_idx]
            if t_idx == 0:
                new_pos = agent_start_goals[agent]['start']
                attractive_force = np.zeros(3)
                repulsive_force = np.zeros(3)
                collision_risk = 0.0
            elif t_idx == n_time - 1:
                new_pos = agent_start_goals[agent]['goal']
                attractive_force = np.zeros(3)
                repulsive_force = np.zeros(3)
                collision_risk = 0.0
            else:
                mask = np.arange(n_agent) != a_idx
                other_positions = current_positions[mask]
                other_vels = current_vels[mask]
                collision_risk = calculate_collision_risk_vec(current_pos, other_positions, R)
                neighbor_center = get_neighbor_center_targets(positions[:, a_idx])[t_idx]
                straight_line_target = get_straight_line_targets(
                    agent_start_goals[agent]['start'],
                    agent_start_goals[agent]['goal'],
                    positions[:, a_idx]
                )[t_idx]
                attractive_force = calculate_combined_attractive_force_vec(
                    current_pos, neighbor_center, straight_line_target,
                    attractive_strength, current_straight_strength, collision_risk
                )
                repulsive_force = calculate_repulsive_force_vec(
                    current_pos, other_positions, current_vel, other_vels, R
                )
                total_force = attractive_force + repulsive_force
                new_pos = current_pos + total_force * alpha
            attractive_magnitude = np.linalg.norm(attractive_force)
            repulsive_magnitude = np.linalg.norm(repulsive_force)
            total_magnitude = np.linalg.norm(attractive_force + repulsive_force)
            modified_paths.append({
                'replan_iter': iteration+1,
                'agent': agent,
                'time': t,
                'x': new_pos[0],
                'y': new_pos[1],
                'z': new_pos[2],
                'collision_risk': collision_risk,
                'attractive_magnitude': attractive_magnitude,
                'repulsive_magnitude': repulsive_magnitude,
                'total_magnitude': total_magnitude,
                'displacement_magnitude': np.linalg.norm(new_pos - current_pos)
            })
            positions[t_idx, a_idx] = new_pos  # 바로 적용
    all_iterations_paths.extend(modified_paths)  # 반복별 결과 누적

# 전체 반복 결과를 하나의 파일로 저장
all_df = pd.DataFrame(all_iterations_paths)
all_df = all_df.sort_values(['replan_iter', 'time', 'agent'])
all_output_path = 'neighbor_center_smoothing_all.csv'
all_df.to_csv(all_output_path, index=False)

print(f"\n전체 반복 경로 결과를 '{all_output_path}'에 저장했습니다.")

print("\n힘 적용 통계(마지막 반복 기준):")
print(f"평균 인력 크기: {all_df[all_df['replan_iter']==n_replanning]['attractive_magnitude'].mean():.4f}")
print(f"평균 척력 크기: {all_df[all_df['replan_iter']==n_replanning]['repulsive_magnitude'].mean():.4f}")
print(f"평균 충돌 위험도: {all_df[all_df['replan_iter']==n_replanning]['collision_risk'].mean():.4f}")
print(f"평균 변위 크기: {all_df[all_df['replan_iter']==n_replanning]['displacement_magnitude'].mean():.4f}")
print(f"총 소요 시간: {time.time() - start_time:.2f}초")
