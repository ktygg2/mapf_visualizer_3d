import pandas as pd
import numpy as np
import time

csv_path = '/home/kty/ros2_ws/src/mapf_visualizer_3d/scripts/result_interpolator.csv'

# 데이터 로드
load_start = time.time()
df = pd.read_csv(csv_path)
data = df[['agent', 'time', 'x', 'y', 'z']].to_numpy(np.float32)
load_end = time.time()
print(f"[데이터 로드] 소요 시간: {load_end - load_start:.2f}초")

# 파라미터
R, safety_margin = 0.5, 0.1
repulsive_threshold = 2 * R + safety_margin
attractive_strength, repulsive_strength = 1.0, 2.0
n_replanning, alpha = 10, 0.3

# 에이전트/시간 정리
time_steps = np.unique(data[:, 1])
agents = np.unique(data[:, 0])
agent_ids = sorted(agents)
n_agents, n_steps = len(agent_ids), len(time_steps)
n_entries = n_agents * n_steps

# 시간 + 에이전트별 위치 캐시
agent_start_goals = {
    agent: (data[data[:, 0] == agent][0, 2:5], data[data[:, 0] == agent][-1, 2:5])
    for agent in agent_ids
}
time_agent_map = {
    t: {a: data[(data[:, 0] == a) & (data[:, 1] == t), 2:5].squeeze() for a in agent_ids}
    for t in time_steps
}

def vectorized_repulsion(positions, threshold, strength):
    n = positions.shape[0]
    disp = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # (n, n, 3)
    distances = np.linalg.norm(disp, axis=2)  # (n, n)
    np.fill_diagonal(distances, np.inf)  # 자기 자신은 제외

    valid = distances < threshold  # (n, n)
    force_magnitudes = np.zeros_like(distances)
    force_magnitudes[valid] = strength * (threshold - distances[valid]) / threshold
    force_magnitudes = force_magnitudes[:, :, np.newaxis]  # (n, n, 1)

    distances_safe = np.where(distances == 0, 1e-6, distances)
    unit_vectors = disp / distances_safe[:, :, np.newaxis]  # (n, n, 3)
    unit_vectors[~valid] = 0  # ✅ 여기를 수정
    return np.sum(unit_vectors * force_magnitudes, axis=1)  # (n, 3)

# 전체 루프
total_start = time.time()
for rep_iter in range(n_replanning):
    print(f"\n=== Replanning {rep_iter+1}/{n_replanning} ===")
    modified = np.zeros((n_entries, 18), np.float32)
    idx = 0

    for t in time_steps:
        step_start = time.time()
        pos_arr = np.array([time_agent_map[t][a] for a in agent_ids], np.float32)  # (n_agents, 3)
        rep_force = vectorized_repulsion(pos_arr, repulsive_threshold, repulsive_strength) if n_agents > 1 else np.zeros_like(pos_arr)

        start_goals = np.array([agent_start_goals[a] for a in agent_ids], np.float32)  # (n, 2, 3)
        line_vecs = start_goals[:, 1] - start_goals[:, 0]
        line_len_sq = np.sum(line_vecs**2, axis=1, keepdims=True) + 1e-8
        proj_t = np.clip(np.sum((pos_arr - start_goals[:, 0]) * line_vecs, axis=1, keepdims=True) / line_len_sq, 0, 1)
        proj_targets = start_goals[:, 0] + proj_t * line_vecs

        # 인력
        disp_attr = proj_targets - pos_arr
        dist_attr = np.linalg.norm(disp_attr, axis=1, keepdims=True)
        attr_force = np.where(dist_attr > 1e-6, (disp_attr / dist_attr) * (attractive_strength * dist_attr), 0)

        total_force = attr_force.squeeze() + rep_force
        new_pos = pos_arr + alpha * total_force

        # 결과 저장
        for i, a in enumerate(agent_ids):
            modified[idx] = [
                a, t,
                *new_pos[i], *pos_arr[i],
                *attr_force[i], *rep_force[i],
                np.linalg.norm(attr_force[i]),
                np.linalg.norm(rep_force[i]),
                np.linalg.norm(total_force[i]),
                np.linalg.norm(new_pos[i] - pos_arr[i])
            ]
            idx += 1

        time_agent_map[t] = {a: new_pos[i] for i, a in enumerate(agent_ids)}
        print(f"  [t={t}] 처리 시간: {time.time() - step_start:.4f}초")

    print(f"재계획 반복 {rep_iter+1} 완료")

print(f"\n전체 실행 시간: {time.time() - total_start:.2f}초")

# 저장
cols = ['agent', 'time', 'x', 'y', 'z',
        'original_x', 'original_y', 'original_z',
        'attractive_force_x', 'attractive_force_y', 'attractive_force_z',
        'repulsive_force_x', 'repulsive_force_y', 'repulsive_force_z',
        'attractive_magnitude', 'repulsive_magnitude',
        'total_magnitude', 'displacement_magnitude']
pd.DataFrame(modified, columns=cols).to_csv('path_smoothing_detailed.csv', index=False)
pd.DataFrame(modified[:, :5], columns=['agent', 'time', 'x', 'y', 'z']).to_csv('path_smoothing_simple.csv', index=False)
