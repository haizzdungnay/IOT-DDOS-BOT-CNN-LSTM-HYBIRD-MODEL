"""
=============================================================================
HISTORY MANAGER - Quản lý lịch sử training và evaluation
=============================================================================
Lưu trữ và truy xuất:
- Training history (thời gian, params, metrics)
- Evaluation results
- Model comparisons
- Reports

Author: IoT Security Research Team
Date: 2026-01-03
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import shutil


class HistoryManager:
    """Quản lý lịch sử training và evaluation"""

    def __init__(self, history_dir: str = "history"):
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)

        # Sub-directories
        self.training_dir = self.history_dir / "training"
        self.evaluation_dir = self.history_dir / "evaluation"
        self.models_dir = self.history_dir / "models"

        for d in [self.training_dir, self.evaluation_dir, self.models_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Index file
        self.index_file = self.history_dir / "index.json"
        self._load_index()

    def _load_index(self):
        """Load index từ file"""
        if self.index_file.exists():
            with open(self.index_file, 'r', encoding='utf-8') as f:
                self.index = json.load(f)
        else:
            self.index = {
                "training_sessions": [],
                "evaluation_sessions": [],
                "last_updated": None
            }

    def _save_index(self):
        """Lưu index ra file"""
        self.index["last_updated"] = datetime.now().isoformat()
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, indent=2, ensure_ascii=False)

    def generate_session_id(self) -> str:
        """Tạo session ID duy nhất"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    # =========================================================================
    # TRAINING HISTORY
    # =========================================================================

    def save_training_session(self,
                              session_id: str,
                              model_name: str,
                              config: dict,
                              history: dict,
                              metrics: dict,
                              model_path: str = None) -> dict:
        """
        Lưu một phiên training

        Args:
            session_id: ID của phiên training
            model_name: Tên model (CNN, LSTM, Hybrid, Parallel)
            config: Cấu hình training (epochs, lr, batch_size, ...)
            history: Training history (loss, acc qua các epochs)
            metrics: Final metrics (accuracy, f1, ...)
            model_path: Đường dẫn đến file model weights

        Returns:
            Session info dict
        """
        session_info = {
            "session_id": session_id,
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "config": config,
            "metrics": metrics,
            "status": "completed"
        }

        # Lưu history chi tiết
        history_file = self.training_dir / f"{session_id}_{model_name}_history.json"
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump({
                "session_info": session_info,
                "history": history
            }, f, indent=2)

        session_info["history_file"] = str(history_file)

        # Copy model weights nếu có
        if model_path and os.path.exists(model_path):
            dest_path = self.models_dir / f"{session_id}_{model_name}.pt"
            shutil.copy2(model_path, dest_path)
            session_info["model_file"] = str(dest_path)

        # Cập nhật index
        self.index["training_sessions"].append(session_info)
        self._save_index()

        return session_info

    def get_training_sessions(self,
                               model_name: str = None,
                               limit: int = None) -> List[dict]:
        """
        Lấy danh sách các phiên training

        Args:
            model_name: Filter theo model (optional)
            limit: Số lượng kết quả tối đa

        Returns:
            List các session info
        """
        sessions = self.index["training_sessions"]

        if model_name:
            sessions = [s for s in sessions if s["model_name"] == model_name]

        # Sắp xếp theo thời gian mới nhất
        sessions = sorted(sessions, key=lambda x: x["timestamp"], reverse=True)

        if limit:
            sessions = sessions[:limit]

        return sessions

    def get_training_history(self, session_id: str, model_name: str) -> Optional[dict]:
        """Lấy chi tiết training history của một phiên"""
        history_file = self.training_dir / f"{session_id}_{model_name}_history.json"

        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    # =========================================================================
    # EVALUATION HISTORY
    # =========================================================================

    def save_evaluation_session(self,
                                 session_id: str,
                                 models_results: Dict[str, dict],
                                 dataset_info: dict = None,
                                 notes: str = None) -> dict:
        """
        Lưu một phiên đánh giá

        Args:
            session_id: ID của phiên đánh giá
            models_results: Dict {model_name: metrics}
            dataset_info: Thông tin về dataset sử dụng
            notes: Ghi chú

        Returns:
            Session info dict
        """
        session_info = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "models": list(models_results.keys()),
            "dataset_info": dataset_info,
            "notes": notes
        }

        # Lưu results chi tiết
        results_file = self.evaluation_dir / f"{session_id}_evaluation.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "session_info": session_info,
                "results": models_results
            }, f, indent=2)

        session_info["results_file"] = str(results_file)

        # Cập nhật index
        self.index["evaluation_sessions"].append(session_info)
        self._save_index()

        return session_info

    def get_evaluation_sessions(self, limit: int = None) -> List[dict]:
        """Lấy danh sách các phiên đánh giá"""
        sessions = self.index["evaluation_sessions"]
        sessions = sorted(sessions, key=lambda x: x["timestamp"], reverse=True)

        if limit:
            sessions = sessions[:limit]

        return sessions

    def get_evaluation_results(self, session_id: str) -> Optional[dict]:
        """Lấy chi tiết kết quả đánh giá"""
        results_file = self.evaluation_dir / f"{session_id}_evaluation.json"

        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def get_latest_evaluation(self) -> Optional[dict]:
        """Lấy kết quả đánh giá mới nhất"""
        sessions = self.get_evaluation_sessions(limit=1)
        if sessions:
            return self.get_evaluation_results(sessions[0]["session_id"])
        return None

    # =========================================================================
    # COMPARISON
    # =========================================================================

    def compare_sessions(self, session_ids: List[str]) -> dict:
        """
        So sánh nhiều phiên đánh giá

        Args:
            session_ids: List các session_id cần so sánh

        Returns:
            Dict chứa so sánh
        """
        comparison = {
            "sessions": [],
            "models": {},
            "metrics": ["accuracy", "precision", "recall", "f1_score", "fpr", "fnr"]
        }

        for sid in session_ids:
            results = self.get_evaluation_results(sid)
            if results:
                comparison["sessions"].append({
                    "session_id": sid,
                    "timestamp": results["session_info"]["timestamp"],
                    "results": results["results"]
                })

                # Aggregate by model
                for model_name, metrics in results["results"].items():
                    if model_name not in comparison["models"]:
                        comparison["models"][model_name] = []
                    comparison["models"][model_name].append({
                        "session_id": sid,
                        "metrics": metrics
                    })

        return comparison

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_statistics(self) -> dict:
        """Lấy thống kê tổng quan"""
        return {
            "total_training_sessions": len(self.index["training_sessions"]),
            "total_evaluation_sessions": len(self.index["evaluation_sessions"]),
            "models_trained": list(set(
                s["model_name"] for s in self.index["training_sessions"]
            )),
            "last_updated": self.index["last_updated"]
        }

    def delete_session(self, session_id: str, session_type: str = "training") -> bool:
        """Xóa một phiên"""
        if session_type == "training":
            sessions = self.index["training_sessions"]
            target_dir = self.training_dir
        else:
            sessions = self.index["evaluation_sessions"]
            target_dir = self.evaluation_dir

        # Tìm và xóa
        for i, s in enumerate(sessions):
            if s["session_id"] == session_id:
                # Xóa files
                for pattern in target_dir.glob(f"{session_id}*"):
                    pattern.unlink()

                # Xóa từ index
                sessions.pop(i)
                self._save_index()
                return True

        return False


# Singleton instance
_history_manager = None

def get_history_manager(history_dir: str = "history") -> HistoryManager:
    """Lấy singleton instance của HistoryManager"""
    global _history_manager
    if _history_manager is None:
        _history_manager = HistoryManager(history_dir)
    return _history_manager
