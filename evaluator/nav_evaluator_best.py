import numpy as np
from fastdtw import fastdtw


class CityNavEvaluator:
    """Episode-level evaluator.

    Required in update(output):
      - output['gt_traj']: list of 3D points
      - output['pred_traj']: list of 3D points

    Optional:
      - any additional numeric scalars (e.g., comm_bytes, uncert_trace_auc, team_success)
        will be aggregated automatically as extra metrics.
    """

    def __init__(self, succ_thresh: float = 20.0):
        self.succ_thresh = float(succ_thresh)
        self.reset()

    def reset(self):
        self.metrics = {"sr": 0.0, "osr": 0.0, "spl": 0.0, "sdtw": 0.0, "ne": 0.0}
        self.extra_metrics = {}
        self.current_status = {
            "gt_traj": [],
            "pred_traj": [],
            "ep_success": 0.0,
            "gt_traj_len": 0.0,
            "pred_traj_len": 0.0,
        }

        self.sr_scorer = []
        self.osr_scorer = []
        self.spl_scorer = []
        self.sdtw_scorer = []
        self.ne_scorer = []

        self._extra_scorers = {}  # key -> list[float]

    def update(self, output: dict):
        self.status_updator(output)
        self.sr_updator()
        self.osr_updator()
        self.ne_updator()
        self.sdtw_updator()
        self.spl_updator()
        self.extra_updator(output)

    def status_updator(self, output: dict):
        updated_status = dict(output)

        gt_traj = np.array(output["gt_traj"], dtype=np.float32)
        pred_traj = np.array(output["pred_traj"], dtype=np.float32)

        ne_final = float(np.linalg.norm(gt_traj[-1] - pred_traj[-1]))
        updated_status["ep_success"] = 1.0 if ne_final < self.succ_thresh else 0.0

        gt_traj_len = float(np.sum(np.linalg.norm(gt_traj[1:] - gt_traj[:-1], axis=1))) if len(gt_traj) >= 2 else 0.0
        pred_traj_len = float(np.sum(np.linalg.norm(pred_traj[1:] - pred_traj[:-1], axis=1))) if len(pred_traj) >= 2 else 0.0

        updated_status["gt_traj_len"] = gt_traj_len
        updated_status["pred_traj_len"] = pred_traj_len
        updated_status["ne_final"] = ne_final

        self.current_status = updated_status

    def sr_updator(self):
        self.sr_scorer.append(float(self.current_status["ep_success"]))

    def osr_updator(self):
        flag = 0.0
        gt_traj = np.array(self.current_status["gt_traj"], dtype=np.float32)
        pred_traj = np.array(self.current_status["pred_traj"], dtype=np.float32)
        goal = gt_traj[-1]
        for i in range(len(pred_traj)):
            if float(np.linalg.norm(pred_traj[i] - goal)) < self.succ_thresh:
                flag = 1.0
                break
        self.osr_scorer.append(flag)

    def ne_updator(self):
        self.ne_scorer.append(float(self.current_status.get("ne_final", 0.0)))

    def sdtw_updator(self):
        gt_traj = np.array(self.current_status["gt_traj"], dtype=np.float32)
        pred_traj = np.array(self.current_status["pred_traj"], dtype=np.float32)
        dtw_dist = float(fastdtw(gt_traj, pred_traj)[0])
        nDTW = float(np.exp(-dtw_dist / (max(1, len(gt_traj)) * self.succ_thresh)))
        sDTW = float(self.current_status["ep_success"]) * nDTW
        self.sdtw_scorer.append(sDTW)

    def spl_updator(self):
        ep_succ = float(self.current_status["ep_success"])
        gt_traj_len = float(self.current_status["gt_traj_len"])
        pred_traj_len = float(self.current_status["pred_traj_len"])
        denom = max(gt_traj_len, pred_traj_len) + 1e-6
        ep_spl = ep_succ * gt_traj_len / denom
        self.spl_scorer.append(float(ep_spl))

    def extra_updator(self, output: dict):
        for k, v in output.items():
            if k in ("gt_traj", "pred_traj"):
                continue
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float, np.integer, np.floating)):
                if k not in self._extra_scorers:
                    self._extra_scorers[k] = []
                self._extra_scorers[k].append(float(v))

    @staticmethod
    def _safe_mean(vals):
        arr = np.array(vals, dtype=np.float32)
        if arr.size == 0:
            return float("nan")
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("nan")
        return float(np.mean(arr))

    def calculate_metrics(self):
        self.metrics["sr"] = self._safe_mean(self.sr_scorer)
        self.metrics["osr"] = self._safe_mean(self.osr_scorer)
        self.metrics["ne"] = self._safe_mean(self.ne_scorer)
        self.metrics["spl"] = self._safe_mean(self.spl_scorer)
        self.metrics["sdtw"] = self._safe_mean(self.sdtw_scorer)
        self.extra_metrics = {k: self._safe_mean(v) for k, v in self._extra_scorers.items()}

    def log_metrics(self, also_extra: bool = True):
        self.calculate_metrics()
        base = (
            f"SR: {self.metrics['sr']:.4f}, SPL: {self.metrics['spl']:.4f}, "
            f"OSR: {self.metrics['osr']:.4f}, SDTW: {self.metrics['sdtw']:.4f}, NE: {self.metrics['ne']:.2f}"
        )
        if also_extra and self.extra_metrics:
            extras = ", ".join([f"{k}: {v:.4f}" for k, v in sorted(self.extra_metrics.items())])
            print(base + " | " + extras)
        else:
            print(base)

    def get_all_metrics(self):
        self.calculate_metrics()
        out = dict(self.metrics)
        out.update(self.extra_metrics)
        return out
