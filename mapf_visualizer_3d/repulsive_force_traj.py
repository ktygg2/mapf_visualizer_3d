import pandas as pd
import numpy as np
import time

# CSV 파일 경로
csv_path = '/home/kty/ros2_ws/src/mapf_visualizer_3d/scripts/result_interpolator.csv'

# 데이터 로드
load_start = time.time()
df = pd.read_csv(csv_path)[['agent', 'time', 'x', 'y', 'z']]
position_data = df.to_numpy(dtype=np.float32)
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

# 고유 에이전트/시간 추출
agents = np.unique(position_data[:, 0])
times = np.unique(position_data[:, 1])
n_agents, n_times = len(agents), len(times)

# 에이전트 매핑 생성
agent_to_idx = {agent: idx for idx, agent in enumerate(agents)}
agent_ids = sorted(agents)

# 위치 배열 초기화 (time, agent, 3)
positions = np.full((n_times, n_agents, 3), np.nan, dtype=np.float32)
for row in position_data:
    a_idx = agent_to_idx[int(row[0])]
    t_idx = np.where(times == row[1])[0][0]
    positions[t_idx, a_idx] = row[2:5]

# 시작/목표 위치 사전 계산
agent_start_goals = {}
for agent in agents:
    agent_data = position_data[position_data[:, 0] == agent]
    agent_start_goals[agent] = (
        agent_data[0, 2:5],   # 시작점
        agent_data[-1, 2:5]    # 목표점
    )

# 완전 벡터화된 척력 계산 함수
def vectorized_repulsion_all(positions, threshold, strength):
    n = positions.shape[0]
    disp = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distances = np.linalg.norm(disp, axis=2)
    np.fill_diagonal(distances, np.inf)
    
    valid = distances < threshold
    force_magnitudes = np.zeros_like(distances)
    force_magnitudes[valid] = strength * (threshold - distances[valid]) / threshold
    force_magnitudes = force_magnitudes[:, :, np.newaxis]
    
    distances_safe = np.where(distances == 0, 1e-6, distances)
    unit_vectors = disp / distances_safe[:, :, np.newaxis]
    unit_vectors[~valid] = 0
    return np.sum(unit_vectors * force_magnitudes, axis=1)

# 결과 저장용 배열 사전 할당
total_entries = n_replanning * n_times * n_agents
results = np.zeros((total_entries, 19), dtype=np.float32)
entry_idx = 0

# 전체 수행
total_start = time.time()
for replan_iter in range(n_replanning):
    print(f"\n=== Replanning Iteration {replan_iter+1}/{n_replanning} ===")
    iter_start = time.time()
    
    for t_idx, t in enumerate(times):
        step_start = time.time()
        current_positions = positions[t_idx].copy()  # (n_agents, 3)
        
        # 1. 척력 벡터 전체 계산
        if n_agents > 1:
            repulsive_forces = vectorized_repulsion_all(
                current_positions, repulsive_threshold, repulsive_strength
            )
        else:
            repulsive_forces = np.zeros_like(current_positions)
        
        # 2. 인력 계산 (벡터화)
        start_goals = np.array([agent_start_goals[a] for a in agent_ids])
        line_vecs = start_goals[:, 1] - start_goals[:, 0]
        line_len_sq = np.sum(line_vecs**2, axis=1, keepdims=True) + 1e-8
        
        current_vecs = current_positions - start_goals[:, 0]
        t_proj = np.clip(
            np.sum(current_vecs * line_vecs, axis=1, keepdims=True) / line_len_sq,
            0.0, 1.0
        )
        projected_targets = start_goals[:, 0] + t_proj * line_vecs
        
        displacement = projected_targets - current_positions
        dist_attr = np.linalg.norm(displacement, axis=1, keepdims=True)
        attractive_force = np.zeros_like(displacement)
        mask = dist_attr.squeeze() > 1e-6
        attractive_force[mask] = (displacement[mask] / dist_attr[mask]) * (attractive_strength * dist_attr[mask])
        
        # 3. 위치 업데이트
        total_force = attractive_force.squeeze() + repulsive_forces
        new_positions = current_positions + alpha * total_force
        positions[t_idx] = new_positions
        
        # 4. 결과 저장 (벡터화)
        for i, agent in enumerate(agent_ids):
            displacement_mag = np.linalg.norm(new_positions[i] - current_positions[i])
            results[entry_idx] = [
                replan_iter+1, agent, t,
                *new_positions[i], *current_positions[i],
                *attractive_force[i], *repulsive_forces[i],
                np.linalg.norm(attractive_force[i]),
                np.linalg.norm(repulsive_forces[i]),
                np.linalg.norm(total_force[i]),
                displacement_mag
            ]
            entry_idx += 1
        
        print(f"  [t={t}] 처리 시간: {time.time() - step_start:.4f}초")
    
    print(f"재계획 반복 {replan_iter+1} 소요 시간: {time.time() - iter_start:.2f}초")

# 최종 결과 저장
cols = [
    'replan_iter', 'agent', 'time', 
    'x', 'y', 'z', 'original_x', 'original_y', 'original_z',
    'attractive_force_x', 'attractive_force_y', 'attractive_force_z',
    'repulsive_force_x', 'repulsive_force_y', 'repulsive_force_z',
    'attractive_magnitude', 'repulsive_magnitude',
    'total_magnitude', 'displacement_magnitude'
]
final_df = pd.DataFrame(results, columns=cols)
final_df = final_df.sort_values(['replan_iter', 'time', 'agent'])
output_path = 'path_smoothing_all.csv'
final_df.to_csv(output_path, index=False)

print(f"\n전체 경로 스무딩 결과를 '{output_path}'에 저장했습니다.")
print(f"전체 실행 시간: {time.time() - total_start:.2f}초")

# 통계 분석
print("\n힘 적용 통계:")
print(f"평균 인력 크기: {final_df['attractive_magnitude'].mean():.4f}")
print(f"평균 척력 크기: {final_df['repulsive_magnitude'].mean():.4f}")
print(f"평균 총 힘 크기: {final_df['total_magnitude'].mean():.4f}")
print(f"평균 변위 크기: {final_df['displacement_magnitude'].mean():.4f}")

# 부가적 분석
significant_smoothing = final_df[final_df['attractive_magnitude'] > 0.1]
collision_avoidance = final_df[final_df['repulsive_magnitude'] > 0.1]
print(f"\n경로 스무딩이 적용된 케이스: {len(significant_smoothing)}개")
print(f"충돌 회피가 적용된 케이스: {len(collision_avoidance)}개")
