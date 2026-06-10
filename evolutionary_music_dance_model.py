
"""
Evolutionary music-dance recommendation model.

The module implements the bachelor's-thesis model described in the conversation:
- personality profile from Q1-Q3;
- current state from stress / energy / music preference / dance preference;
- 25 abstract strategies over a 5x5 grid;
- base compatibility score;
- empirical effectiveness estimate;
- evolutionary selection using a discrete replicator dynamic;
- catalog layer that turns an abstract strategy into a concrete recommendation.

The code is intentionally written to be readable and modifiable for research use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


PersonalityType = str


def _validate_scale_1_5(value: int, name: str) -> int:
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer in the range [1, 5].")
    if value < 1 or value > 5:
        raise ValueError(f"{name} must be in the range [1, 5]. Got {value!r}.")
    return int(value)


def _stable_softmax(values: np.ndarray, beta: float) -> np.ndarray:
    if values.size == 0:
        return values
    shifted = values - np.max(values)
    weights = np.exp(beta * shifted)
    total = np.sum(weights)
    if total <= 0:
        return np.full_like(weights, fill_value=1.0 / len(weights), dtype=float)
    return weights / total


@dataclass(frozen=True)
class Strategy:
    """Abstract strategy: music intensity + dance/social intensity."""
    strategy_id: int
    music_level: int
    dance_level: int
    label: str

    @property
    def activation(self) -> float:
        return (self.music_level + self.dance_level) / 2.0

    @property
    def display_id(self) -> int:
        return self.strategy_id + 1


@dataclass
class ModelConfig:
    """Parameters of the evolutionary recommender."""
    w_personality: float = 0.25
    w_activation: float = 0.30
    w_preferences: float = 0.30
    w_safety: float = 0.15

    alpha_energy: float = 0.5
    alpha_stress: float = 0.5
    personality_bias: Dict[PersonalityType, float] = field(
        default_factory=lambda: {"I": -0.5, "A": 0.0, "E": 0.5}
    )

    reward_weight_stress: float = 0.60
    reward_weight_improvement: float = 0.40

    theta: float = 0.65
    beta: float = 4.0
    epsilon: float = 1e-6
    exploration_rate: float = 0.10

    admissible_tolerance_music: int = 2
    admissible_tolerance_dance: int = 2

    def __post_init__(self) -> None:
        base_sum = (
            self.w_personality
            + self.w_activation
            + self.w_preferences
            + self.w_safety
        )
        reward_sum = self.reward_weight_stress + self.reward_weight_improvement
        if not np.isclose(base_sum, 1.0):
            raise ValueError(
                "Base-score weights must sum to 1.0. "
                f"Current sum: {base_sum:.6f}"
            )
        if not np.isclose(reward_sum, 1.0):
            raise ValueError(
                "Reward weights must sum to 1.0. "
                f"Current sum: {reward_sum:.6f}"
            )
        if not (0.0 < self.theta < 1.0):
            raise ValueError("theta must be in the open interval (0, 1).")
        if self.beta <= 0:
            raise ValueError("beta must be positive.")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        if not (0.0 <= self.exploration_rate <= 1.0):
            raise ValueError("exploration_rate must be in [0, 1].")


@dataclass(frozen=True)
class UserProfile:
    q1: int
    q2: int
    q3: int
    personality_index: float
    personality_type: PersonalityType


@dataclass(frozen=True)
class SessionInput:
    personality_type: PersonalityType
    stress: int
    energy: int
    music_preference: int
    dance_preference: int

    def __post_init__(self) -> None:
        if self.personality_type not in {"I", "A", "E"}:
            raise ValueError(
                "personality_type must be one of {'I', 'A', 'E'}."
            )
        _validate_scale_1_5(self.stress, "stress")
        _validate_scale_1_5(self.energy, "energy")
        _validate_scale_1_5(self.music_preference, "music_preference")
        _validate_scale_1_5(self.dance_preference, "dance_preference")


@dataclass
class Recommendation:
    strategy: Strategy
    distribution: np.ndarray
    base_scores: np.ndarray
    fitness: np.ndarray
    admissible_mask: np.ndarray
    components: Dict[str, np.ndarray]
    desired_activation: float
    catalog_item: Dict[str, Any]
    used_exploration: bool
    top_candidates: List[Tuple[int, float]]

    @property
    def strategy_id(self) -> int:
        return self.strategy.strategy_id

    @property
    def probability(self) -> float:
        return float(self.distribution[self.strategy_id])


@dataclass
class UserState:
    profile: UserProfile
    distribution: Optional[np.ndarray] = None
    average_rewards: np.ndarray = field(
        default_factory=lambda: np.full(25, 0.5, dtype=float)
    )
    counts: np.ndarray = field(
        default_factory=lambda: np.zeros(25, dtype=int)
    )
    history: List[Dict[str, Any]] = field(default_factory=list)
    catalog_cursor: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.average_rewards = np.asarray(self.average_rewards, dtype=float)
        self.counts = np.asarray(self.counts, dtype=int)
        if self.average_rewards.shape != (25,):
            raise ValueError("average_rewards must have shape (25,).")
        if self.counts.shape != (25,):
            raise ValueError("counts must have shape (25,).")
        if self.distribution is not None:
            self.distribution = np.asarray(self.distribution, dtype=float)
            if self.distribution.shape != (25,):
                raise ValueError("distribution must have shape (25,).")


class EvolutionaryMusicDanceModel:
    """
    Recommender implementing the evolutionary model from the thesis chapter.

    Workflow:
    1. Create a user profile from Q1-Q3.
    2. Create a UserState from that profile.
    3. Build a SessionInput from Q4-Q7.
    4. Call recommend(...) to get a strategy and a catalog recommendation.
    5. After the session, call register_feedback(...) with stress_after and
       overall_improvement.
    6. Repeat for the next session.
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        seed: Optional[int] = None,
        catalog: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    ) -> None:
        self.config = config or ModelConfig()
        self.rng = np.random.default_rng(seed)

        self.strategies: List[Strategy] = self._build_strategies()
        self.music_levels = np.array(
            [strategy.music_level for strategy in self.strategies],
            dtype=float,
        )
        self.dance_levels = np.array(
            [strategy.dance_level for strategy in self.strategies],
            dtype=float,
        )
        self.activations = (self.music_levels + self.dance_levels) / 2.0

        self.strategy_lookup: Dict[Tuple[int, int], int] = {
            (s.music_level, s.dance_level): s.strategy_id for s in self.strategies
        }

        self.catalog = catalog if catalog is not None else self._build_default_catalog()

    # ------------------------------------------------------------------
    # User profile and session helpers
    # ------------------------------------------------------------------
    @staticmethod
    def personality_index(q1: int, q2: int, q3: int) -> float:
        q1 = _validate_scale_1_5(q1, "q1")
        q2 = _validate_scale_1_5(q2, "q2")
        q3 = _validate_scale_1_5(q3, "q3")
        return float((q1 + q2 + q3) / 3.0)

    @staticmethod
    def classify_personality(index: float) -> PersonalityType:
        if 1.0 <= index <= 2.49:
            return "I"
        if 2.50 <= index <= 3.49:
            return "A"
        if 3.50 <= index <= 5.0:
            return "E"
        raise ValueError(
            f"personality index must be in [1.0, 5.0]. Got {index!r}."
        )

    def build_user_profile(self, q1: int, q2: int, q3: int) -> UserProfile:
        p_index = self.personality_index(q1, q2, q3)
        return UserProfile(
            q1=q1,
            q2=q2,
            q3=q3,
            personality_index=p_index,
            personality_type=self.classify_personality(p_index),
        )

    def create_user_state(self, profile: UserProfile) -> UserState:
        return UserState(profile=profile)

    def make_session(
        self,
        personality_type: PersonalityType,
        stress: int,
        energy: int,
        music_preference: int,
        dance_preference: int,
    ) -> SessionInput:
        return SessionInput(
            personality_type=personality_type,
            stress=stress,
            energy=energy,
            music_preference=music_preference,
            dance_preference=dance_preference,
        )

    # ------------------------------------------------------------------
    # Strategy space and catalog
    # ------------------------------------------------------------------
    def _build_strategies(self) -> List[Strategy]:
        music_labels = {
            1: "много спокойна музика",
            2: "спокойна музика",
            3: "умерена музика",
            4: "ритмична музика",
            5: "силно енергична музика",
        }
        dance_labels = {
            1: "почти без движение",
            2: "плавни самостоятелни движения",
            3: "свободен индивидуален танц",
            4: "танц с партньор или малка група",
            5: "енергичен групов танц",
        }

        strategies: List[Strategy] = []
        strategy_id = 0
        for music_level in range(1, 6):
            for dance_level in range(1, 6):
                strategies.append(
                    Strategy(
                        strategy_id=strategy_id,
                        music_level=music_level,
                        dance_level=dance_level,
                        label=(
                            f"{music_labels[music_level]} + "
                            f"{dance_labels[dance_level]}"
                        ),
                    )
                )
                strategy_id += 1
        return strategies

    def _build_default_catalog(self) -> Dict[int, List[Dict[str, Any]]]:
        music_styles = {
            1: ["ambient", "тиха инструментална", "нежна пиано музика"],
            2: ["акустична", "софт поп", "мек джаз"],
            3: ["лоу-фай", "умерен поп", "фънк с умерено темпо"],
            4: ["ритмичен поп", "електро суинг", "динамичен латино ритъм"],
            5: ["денс", "енергично електро", "силно ритмичен фюжън"],
        }
        movement_styles = {
            1: [
                "слушане с дихателно отпускане",
                "седящо отпускане с минимални движения",
                "кратка релаксация без танц",
            ],
            2: [
                "бавни разтягания",
                "плавни движения на ръце и рамене",
                "лека самостоятелна двигателна рутина",
            ],
            3: [
                "свободен индивидуален танц",
                "кратка ритмична импровизация",
                "домашна танцова сесия",
            ],
            4: [
                "танц по двойки",
                "малка групова хореография",
                "социален танц в ограничен кръг",
            ],
            5: [
                "енергичен групов танц",
                "динамична групова хореография",
                "интензивна танцова активност в група",
            ],
        }
        duration_map = {
            1: 6,
            2: 8,
            3: 10,
            4: 12,
            5: 15,
        }

        catalog: Dict[int, List[Dict[str, Any]]] = {}
        for strategy in self.strategies:
            entries: List[Dict[str, Any]] = []
            for i in range(3):
                music_text = music_styles[strategy.music_level][i]
                move_text = movement_styles[strategy.dance_level][i]
                duration = duration_map[strategy.dance_level] + strategy.music_level - 1
                title = f"{music_text.title()} + {move_text}"
                description = (
                    f"Комбинация от {music_text} и {move_text}, "
                    f"подходяща за музикално ниво {strategy.music_level} "
                    f"и двигателно ниво {strategy.dance_level}."
                )
                entries.append(
                    {
                        "title": title,
                        "description": description,
                        "music_level": strategy.music_level,
                        "dance_level": strategy.dance_level,
                        "duration_minutes": duration,
                        "tags": [music_text, move_text],
                    }
                )
            catalog[strategy.strategy_id] = entries
        return catalog

    # ------------------------------------------------------------------
    # Core model functions
    # ------------------------------------------------------------------
    @staticmethod
    def personality_center(personality_type: PersonalityType) -> Tuple[float, float]:
        centers = {"I": (2.0, 2.0), "A": (3.0, 3.0), "E": (4.0, 4.0)}
        if personality_type not in centers:
            raise ValueError(f"Unknown personality type: {personality_type!r}")
        return centers[personality_type]

    def desired_activation(self, session: SessionInput) -> float:
        raw_value = (
            3.0
            + self.config.alpha_energy * (session.energy - 3)
            - self.config.alpha_stress * (session.stress - 3)
            + self.config.personality_bias[session.personality_type]
        )
        return float(np.clip(raw_value, 1.0, 5.0))

    def admissible_mask(self, session: SessionInput) -> np.ndarray:
        return (
            (np.abs(self.music_levels - session.music_preference) <= self.config.admissible_tolerance_music)
            & (np.abs(self.dance_levels - session.dance_preference) <= self.config.admissible_tolerance_dance)
        )

    def base_components(self, session: SessionInput) -> Dict[str, np.ndarray]:
        mu_c, delta_c = self.personality_center(session.personality_type)

        personality_match = 1.0 - (
            np.abs(self.music_levels - mu_c) + np.abs(self.dance_levels - delta_c)
        ) / 8.0

        target_activation = self.desired_activation(session)
        activation_match = 1.0 - np.abs(self.activations - target_activation) / 4.0

        preference_match = 1.0 - (
            np.abs(self.music_levels - session.music_preference)
            + np.abs(self.dance_levels - session.dance_preference)
        ) / 8.0

        safety = 1.0 - np.maximum(0.0, self.activations - target_activation) / 4.0

        return {
            "personality": np.clip(personality_match, 0.0, 1.0),
            "activation": np.clip(activation_match, 0.0, 1.0),
            "preferences": np.clip(preference_match, 0.0, 1.0),
            "safety": np.clip(safety, 0.0, 1.0),
        }

    def base_score(
        self, session: SessionInput
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        components = self.base_components(session)
        score = (
            self.config.w_personality * components["personality"]
            + self.config.w_activation * components["activation"]
            + self.config.w_preferences * components["preferences"]
            + self.config.w_safety * components["safety"]
        )
        return np.clip(score, 0.0, 1.0), components

    def fitness(
        self,
        session: SessionInput,
        state: UserState,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        base_scores, components = self.base_score(session)
        admissible = self.admissible_mask(session)
        fit = (
            self.config.theta * base_scores
            + (1.0 - self.config.theta) * state.average_rewards
            + self.config.epsilon
        )
        fit = np.where(admissible, fit, 0.0)
        return fit, base_scores, components, admissible

    def _initialize_distribution(
        self, base_scores: np.ndarray, admissible: np.ndarray
    ) -> np.ndarray:
        x = np.zeros(len(self.strategies), dtype=float)
        admissible_idx = np.flatnonzero(admissible)
        if admissible_idx.size == 0:
            raise RuntimeError("No admissible strategies available.")
        probs = _stable_softmax(base_scores[admissible_idx], beta=self.config.beta)
        x[admissible_idx] = probs
        return x

    @staticmethod
    def _project_distribution(distribution: np.ndarray, admissible: np.ndarray) -> np.ndarray:
        x = np.where(admissible, distribution, 0.0).astype(float)
        total = float(np.sum(x))
        if total <= 0.0:
            count = int(np.sum(admissible))
            if count == 0:
                raise RuntimeError("Cannot project onto an empty admissible set.")
            x[admissible] = 1.0 / count
            return x
        return x / total

    def _replicator_update(
        self,
        distribution: np.ndarray,
        fitness: np.ndarray,
        admissible: np.ndarray,
    ) -> np.ndarray:
        x_prev = self._project_distribution(distribution, admissible)
        denom = float(np.dot(x_prev, fitness))
        if denom <= 0.0:
            return self._project_distribution(np.zeros_like(x_prev), admissible)
        x_new = np.zeros_like(x_prev)
        x_new[admissible] = x_prev[admissible] * fitness[admissible] / denom
        return self._project_distribution(x_new, admissible)

    def preview_distribution(
        self, state: UserState, session: SessionInput
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """
        Compute the distribution that would be used for the current session
        without modifying the UserState.
        """
        fit, base_scores, components, admissible = self.fitness(session, state)
        if state.distribution is None:
            x_new = self._initialize_distribution(base_scores, admissible)
        else:
            x_new = self._replicator_update(state.distribution, fit, admissible)
        return x_new, base_scores, fit, components, admissible

    def _select_strategy_id(
        self, distribution: np.ndarray, admissible: np.ndarray
    ) -> Tuple[int, bool]:
        admissible_idx = np.flatnonzero(admissible)
        if admissible_idx.size == 0:
            raise RuntimeError("No admissible strategy can be selected.")

        if admissible_idx.size == 1:
            return int(admissible_idx[0]), False

        if self.rng.random() < self.config.exploration_rate:
            probabilities = distribution[admissible_idx]
            probabilities = probabilities / probabilities.sum()
            chosen = int(self.rng.choice(admissible_idx, p=probabilities))
            return chosen, True

        best_value = float(np.max(distribution[admissible_idx]))
        best_idx = admissible_idx[np.isclose(distribution[admissible_idx], best_value)]
        chosen = int(self.rng.choice(best_idx))
        return chosen, False

    def _next_catalog_item(self, state: UserState, strategy_id: int) -> Dict[str, Any]:
        items = self.catalog[strategy_id]
        cursor = state.catalog_cursor.get(strategy_id, 0)
        item = items[cursor % len(items)]
        state.catalog_cursor[strategy_id] = (cursor + 1) % len(items)
        return item

    def recommend(self, state: UserState, session: SessionInput) -> Recommendation:
        distribution, base_scores, fit, components, admissible = self.preview_distribution(
            state, session
        )
        state.distribution = distribution.copy()

        strategy_id, used_exploration = self._select_strategy_id(distribution, admissible)
        catalog_item = self._next_catalog_item(state, strategy_id)

        top_indices = np.argsort(distribution)[::-1][:5]
        top_candidates = [
            (int(idx), float(distribution[idx]))
            for idx in top_indices
            if admissible[idx]
        ]

        return Recommendation(
            strategy=self.strategies[strategy_id],
            distribution=distribution.copy(),
            base_scores=base_scores.copy(),
            fitness=fit.copy(),
            admissible_mask=admissible.copy(),
            components={k: v.copy() for k, v in components.items()},
            desired_activation=self.desired_activation(session),
            catalog_item=catalog_item,
            used_exploration=used_exploration,
            top_candidates=top_candidates,
        )

    # ------------------------------------------------------------------
    # Feedback and learning
    # ------------------------------------------------------------------
    def compute_reward(
        self,
        stress_before: int,
        stress_after: int,
        overall_improvement: int,
    ) -> float:
        stress_before = _validate_scale_1_5(stress_before, "stress_before")
        stress_after = _validate_scale_1_5(stress_after, "stress_after")
        overall_improvement = _validate_scale_1_5(
            overall_improvement, "overall_improvement"
        )

        delta_stress = max(0, stress_before - stress_after) / 4.0
        subjective_gain = (overall_improvement - 1) / 4.0

        reward = (
            self.config.reward_weight_stress * delta_stress
            + self.config.reward_weight_improvement * subjective_gain
        )
        return float(np.clip(reward, 0.0, 1.0))

    def register_feedback(
        self,
        state: UserState,
        recommendation: Recommendation,
        session: SessionInput,
        stress_after: int,
        overall_improvement: int,
    ) -> float:
        reward = self.compute_reward(
            stress_before=session.stress,
            stress_after=stress_after,
            overall_improvement=overall_improvement,
        )

        idx = recommendation.strategy_id
        count = int(state.counts[idx])
        previous_average = float(state.average_rewards[idx])
        new_average = (count * previous_average + reward) / (count + 1)

        state.average_rewards[idx] = new_average
        state.counts[idx] += 1

        state.history.append(
            {
                "session_no": len(state.history) + 1,
                "personality_type": session.personality_type,
                "stress_before": session.stress,
                "energy": session.energy,
                "music_preference": session.music_preference,
                "dance_preference": session.dance_preference,
                "desired_activation": recommendation.desired_activation,
                "strategy_id": idx,
                "strategy_display_id": recommendation.strategy.display_id,
                "strategy_label": recommendation.strategy.label,
                "music_level": recommendation.strategy.music_level,
                "dance_level": recommendation.strategy.dance_level,
                "probability": recommendation.probability,
                "base_score": float(recommendation.base_scores[idx]),
                "fitness": float(recommendation.fitness[idx]),
                "catalog_title": recommendation.catalog_item["title"],
                "catalog_description": recommendation.catalog_item["description"],
                "used_exploration": recommendation.used_exploration,
                "stress_after": _validate_scale_1_5(stress_after, "stress_after"),
                "overall_improvement": _validate_scale_1_5(
                    overall_improvement, "overall_improvement"
                ),
                "reward": reward,
                "strategy_avg_reward_after_update": float(state.average_rewards[idx]),
                "strategy_count_after_update": int(state.counts[idx]),
            }
        )
        return reward

    # ------------------------------------------------------------------
    # Data-frame helpers for analysis
    # ------------------------------------------------------------------
    def recommendation_dataframe(self, recommendation: Recommendation) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for strategy in self.strategies:
            idx = strategy.strategy_id
            rows.append(
                {
                    "strategy_id": idx,
                    "strategy_display_id": strategy.display_id,
                    "music_level": strategy.music_level,
                    "dance_level": strategy.dance_level,
                    "activation": strategy.activation,
                    "label": strategy.label,
                    "admissible": bool(recommendation.admissible_mask[idx]),
                    "probability": float(recommendation.distribution[idx]),
                    "base_score": float(recommendation.base_scores[idx]),
                    "fitness": float(recommendation.fitness[idx]),
                    "personality_component": float(
                        recommendation.components["personality"][idx]
                    ),
                    "activation_component": float(
                        recommendation.components["activation"][idx]
                    ),
                    "preference_component": float(
                        recommendation.components["preferences"][idx]
                    ),
                    "safety_component": float(
                        recommendation.components["safety"][idx]
                    ),
                }
            )
        df = pd.DataFrame(rows)
        return df.sort_values(
            by=["admissible", "probability", "base_score"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    @staticmethod
    def history_dataframe(state: UserState) -> pd.DataFrame:
        if not state.history:
            return pd.DataFrame()
        return pd.DataFrame(state.history)

    def strategy_statistics_dataframe(self, state: UserState) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for strategy in self.strategies:
            idx = strategy.strategy_id
            rows.append(
                {
                    "strategy_id": idx,
                    "strategy_display_id": strategy.display_id,
                    "label": strategy.label,
                    "music_level": strategy.music_level,
                    "dance_level": strategy.dance_level,
                    "activation": strategy.activation,
                    "count": int(state.counts[idx]),
                    "average_reward": float(state.average_rewards[idx]),
                    "current_probability": (
                        float(state.distribution[idx])
                        if state.distribution is not None
                        else np.nan
                    ),
                }
            )
        return pd.DataFrame(rows).sort_values(
            by=["average_reward", "count", "current_probability"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Convenience utilities
    # ------------------------------------------------------------------
    def questionnaire_dataframe(self, survey_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add personality_index and personality_type columns to a DataFrame
        containing Q1, Q2, Q3 columns.
        """
        required = {"Q1", "Q2", "Q3"}
        missing = required.difference(survey_df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns for questionnaire processing: {sorted(missing)}"
            )

        output = survey_df.copy()
        output["personality_index"] = output[["Q1", "Q2", "Q3"]].mean(axis=1)
        output["personality_type"] = output["personality_index"].apply(
            self.classify_personality
        )
        return output

    def describe_personality(self, personality_type: PersonalityType) -> str:
        descriptions = {
            "I": "интроверт",
            "A": "амбиверт",
            "E": "екстроверт",
        }
        return descriptions[personality_type]

    def strategy_from_levels(self, music_level: int, dance_level: int) -> Strategy:
        music_level = _validate_scale_1_5(music_level, "music_level")
        dance_level = _validate_scale_1_5(dance_level, "dance_level")
        idx = self.strategy_lookup[(music_level, dance_level)]
        return self.strategies[idx]


__all__ = [
    "Strategy",
    "ModelConfig",
    "UserProfile",
    "UserState",
    "SessionInput",
    "Recommendation",
    "EvolutionaryMusicDanceModel",
]


if __name__ == "__main__":
    model = EvolutionaryMusicDanceModel(seed=42)
    profile = model.build_user_profile(q1=2, q2=2, q3=3)
    state = model.create_user_state(profile)

    session = model.make_session(
        personality_type=profile.personality_type,
        stress=4,
        energy=2,
        music_preference=2,
        dance_preference=2,
    )

    recommendation = model.recommend(state, session)
    reward = model.register_feedback(
        state=state,
        recommendation=recommendation,
        session=session,
        stress_after=2,
        overall_improvement=4,
    )

    print("=== Demo run of the evolutionary music-dance model ===")
    print("Personality type:", model.describe_personality(profile.personality_type))
    print("Recommended strategy:", recommendation.strategy.label)
    print("Catalog item:", recommendation.catalog_item["title"])
    print("Reward after feedback:", round(reward, 3))
