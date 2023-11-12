"""
Модель ядра кредитного скоринга.

Возвращает решение о выдаче кредите и одобренную сумму, в случае одобрения.
"""
import sys

import joblib
import pandas as pd
from api import Features, ScoringDecision, ScoringResult
from calculator import Calculator

sys.path.extend(['../../../config', '../../app/utils'])


class AdvancedModel(object):
    """Класс для моделей c расчетом proba и threshold."""

    approve_threshold = 0.25

    def __init__(self, model_path: str):
        """
        Создает объект класса.

        Parameters:
        - model_path (str): Путь до модели
        """
        self._model = joblib.load(model_path)
        self._calculator = Calculator()

    def get_scoring_result(self, features):
        """Функция для получения результатов скоринга.

        features: Признаки для предсказания вероятности дефолта

        Returns:
        - ScoringResult (class): Объект класса с результатом скоринга
        """
        proba = self._predict_proba(Features(**features))[0]

        final_decision = ScoringDecision.declined
        final_amount = 0

        if proba < self.approve_threshold:
            final_score = self._calculator.score_loan_amt(features)
            final_decision = ScoringDecision.accepted
            final_amount = self._calculator.calc_amount(proba, final_score)

        return ScoringResult(
            decision=final_decision,
            amount=final_amount,
            threshold=self.approve_threshold,
            proba=proba,
        )

    def _predict_proba(self, features: Features) -> float:
        """Определяет вероятность дефолта по займу."""
        features = pd.DataFrame([features.__dict__])
        return self._model.predict_proba(features)[:, 1]
