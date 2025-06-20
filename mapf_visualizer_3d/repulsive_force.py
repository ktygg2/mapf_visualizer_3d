import pandas as pd
import numpy as np
import time
import os

# CSV 파일 경로
csv_path = '/home/kty/ros2_ws/src/mapf_visualizer_3d/scripts/result_interpolator.csv'

# 데이터 로드 및 전처리
load_start = time.time()
interpolated_points = pd.read_csv(csv_path)
position_data = interpolated_points[['agent', 'time', 'x', 'y', 'z']].to_numpy().astype(np.float32)
load_end = time.time()
print(f"[데이터 로드] 소요 시간: {load_end - load_start:.2f}초")

# 파라미터 설정
R = 0.5
safety_margin = 0.1
attractive_strength = 1.0
repulsive_strength = 2.0
repulsive_threshold = 2 * R + safety_margin
n_replanning = 10
alpha = 0.3

# 데이터 구조 최적화
time_steps = np.unique(position_data[:, 1])
agents = np.unique(position_data[:, 0])

# 시간별/에이전트별 위치 매핑 생성
time_agent_map = {}
for t in time_steps:
    time_mask = position_data[:, 1] == t
    time_agent_map[t] = {
        agent: position_data[time_mask & (position_data[:, 0] == agent), 2:5].squeeze()
        for agent in agents
    }

# 시작점/목표점 사전 계산
agent_start_goals = {}
for agent in agents:
    agent_mask = position_data[:, 0] == agent
    agent_data = position_data[agent_mask]
    agent_start_goals[agent] = (
        agent_data[0, 2:5],  # 시작점
        agent_data[-1, 2:5]  # 목표점
    )

def vectorized_repulsion(current_pos, other_positions, threshold, strength):
    """벡터화된 척력 계산 함수"""
    displacements = current_pos - other_positions
    distances = np.linalg.norm(displacements, axis=1, keepdims=True)
    distances = np.clip(distances, 1e-6, None)  # 0으로 나누기 방지
    
    valid = distances < threshold
    force_magnitudes = strength * (threshold - distances) / threshold * valid
    unit_vectors = displacements / distances
    return np.sum(unit_vectors * force_magnitudes, axis=0)

total_start = time.time()

for replan_iter in range(n_replanning):
    print(f"\n=== Replanning Iteration {replan_iter+1} / {n_replanning} ===")
    modified_paths = []
    iter_start = time.time()
    
    for t in time_steps:
        step_start = time.time()
        current_agents = time_agent_map[t]
        agent_ids = list(current_agents.keys())
        positions = np.array([current_agents[a] for a in agent_ids])
        
        for idx, agent in enumerate(agent_ids):
            current_pos = positions[idx]
            start, goal = agent_start_goals[agent]
            
            # 투영점 계산 (벡터화)
            line_vec = goal - start
            line_len = np.linalg.norm(line_vec)
            if line_len > 1e-6:
                current_vec = current_pos - start
                t_proj = np.dot(current_vec, line_vec) / (line_len ** 2)
                t_proj = np.clip(t_proj, 0.0, 1.0)
                projected_target = start + t_proj * line_vec
            else:
                projected_target = current_pos.copy()
            
            # 인력 계산
            displacement = projected_target - current_pos
            distance = np.linalg.norm(displacement)
            if distance > 0:
                attractive_force = (displacement / distance) * (attractive_strength * distance)
            else:
                attractive_force = np.zeros(3)
            
            # 척력 계산 (벡터화)
            if len(positions) > 1:
                other_positions = np.delete(positions, idx, axis=0)
                repulsive_force = vectorized_repulsion(current_pos, other_positions, 
                                                     repulsive_threshold, repulsive_strength)
            else:
                repulsive_force = np.zeros(3)
            
            # 위치 업데이트
            new_pos = current_pos + (attractive_force + repulsive_force) * alpha
            
            # 결과 저장 (NumPy 배열로 최적화)
            modified_paths.append([
                agent, t, 
                new_pos[0], new_pos[1], new_pos[2],
                current_pos[0], current_pos[1], current_pos[2],
                attractive_force[0], attractive_force[1], attractive_force[2],
                repulsive_force[0], repulsive_force[1], repulsive_force[2],
                np.linalg.norm(attractive_force),
                np.linalg.norm(repulsive_force),
                np.linalg.norm(attractive_force + repulsive_force),
                np.linalg.norm(new_pos - current_pos)
            ])
        
        step_end = time.time()
        print(f"  [t={t}] 처리 시간: {step_end - step_start:.4f}초")
    
    # 데이터 구조 갱신
    modified_arr = np.array(modified_paths, dtype=np.float32)
    
    # 다음 반복을 위한 데이터 업데이트
    for t in time_steps:
        time_mask = modified_arr[:, 1] == t
        time_agent_map[t] = {
            agent: modified_arr[time_mask & (modified_arr[:, 0] == agent), 2:5].squeeze()
            for agent in agents
        }
    
    iter_end = time.time()
    print(f"재계획 반복 {replan_iter+1} 소요 시간: {iter_end - iter_start:.2f}초")

total_end = time.time()
print(f"\n전체 실행 시간: {total_end - total_start:.2f}초")

# 결과 저장
result_df = pd.DataFrame(modified_paths, columns=[
    'agent', 'time', 'x', 'y', 'z',
    'original_x', 'original_y', 'original_z',
    'attractive_force_x', 'attractive_force_y', 'attractive_force_z',
    'repulsive_force_x', 'repulsive_force_y', 'repulsive_force_z',
    'attractive_magnitude', 'repulsive_magnitude',
    'total_magnitude', 'displacement_magnitude'
])
result_df.to_csv('path_smoothing_detailed.csv', index=False)
result_df[['agent', 'time', 'x', 'y', 'z']].to_csv('path_smoothing_simple.csv', index=False)
