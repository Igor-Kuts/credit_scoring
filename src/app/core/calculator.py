import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '../../../..'))

from src.app.core.api import Features, LoanAmount


class Calculator(object):
    """Класс для подсчёта суммы выдаваемого кредита."""

    _scores_list = [0, 1, 3, 5]
    _decision_brdrs = [7, 25, 30]
    _safe_proba = 0.08
    
    def calc_amount(self, proba: float, final_score: int) -> int:
        """
        Рассчитывает размер выдаваемого займа на основе вероятности дефолта 
        и итоговой кредитной оценки.

        Эта функция определяет окончательный размер выдаваемого займа, 
        используя метод score_loan_amt, основываясь на следующих критериях:
        - Возраст заемщика.
        - Трудовая история заемщика.
        - Доход заемщика.
        - Цель кредита.
        - Информация из внешних источников.
        - Наличие микрозаймов.

        Параметры:
        proba (float): Вероятность дефолта для заемщика.
        final_score (int): Кредитная оценка, рассчитанная на основе характеристик заемщика.

        Возвращает:
        int: Рассчитанный размер займа, хранящийся в LoanAmount.
            Возвращаемые значения соответствуют одному из следующих условий:
            - LoanAmount.max_loan: Если вероятность дефолта (proba) <= _safe_proba.
            - LoanAmount.max_loan: Если final_score больше или равно 25.
            - LoanAmount.high_loan: Если final_score находится в диапазоне [25, 30).
            - LoanAmount.mid_loan: Если final_score находится в диапазоне [7, 25).
            - LoanAmount.min_loan: Если final_score меньше 7.
        """
        loan_amount = LoanAmount()
        if proba <= self._safe_proba: 
            return loan_amount.max_loan
        if final_score >= self._decision_brdrs[2]: 
            return loan_amount.max_loan
        if self._decision_brdrs[1] <= final_score < self._decision_brdrs[2]: 
            return loan_amount.high_loan
        if self._decision_brdrs[0] <= final_score < self._decision_brdrs[1]: 
            return loan_amount.mid_loan
        if self._decision_brdrs[0] > final_score: 
            return loan_amount.min_loan

    def score_loan_amt(self, features: Features) -> int:
        """Расчёт финального score для клиента для определения суммы кредита."""
        final_score = 0
        final_score += self._age_score(features)
        final_score += self._emp_days_score(features)
        final_score += self._income_score(features)
        final_score += self._credit_type_score(features)
        final_score += self._external2_score(features)
        final_score += self._microloan_score(features)

        return final_score

    # Оценка возраста заёмщика
    def _age_score(self, features: Features) -> int:

        min_age = 26
        mid_age = 50
        max_age = 65

        points = 0
        if features.age_years < min_age or features.age_years > max_age:
            points = self._scores_list[0]
        elif min_age <= features.age_years <= mid_age:
            points = self._scores_list[3]
        elif mid_age <= features.age_years <= max_age:
            points = self._scores_list[2]

        return points

    # Оценка трудового стажа заёмщика
    def _emp_days_score(self, features: Features) -> int:

        min_emp_days = 365
        mid_emp_days = 1095
        max_emp_days = 1825

        points = 0
        if features.days_employed < min_emp_days:
            points = self._scores_list[0]
        elif min_emp_days < features.days_employed <= mid_emp_days:
            points = self._scores_list[1]
        elif mid_emp_days < features.days_employed <= max_emp_days:
            points = self._scores_list[2]
        elif features.days_employed > max_emp_days:
            points = self._scores_list[3]

        return points

    # Оценка заработка заёщика по медианному заработку
    def _income_score(self, features: Features) -> int:

        points = 0
        median_inc = 150_000
        coeff = 1.5

        if features.amt_income_total >= median_inc * coeff:
            points = self._scores_list[3]
        elif features.amt_income_total >= median_inc:
            points = self._scores_list[2]
        elif (median_inc / 2) < features.amt_income_total < median_inc:
            points = self._scores_list[1]
        elif median_inc / 2 > features.amt_income_total:
            points = self._scores_list[0]

        return points

    # Оценка по цели займа
    def _credit_type_score(self, features: Features) -> int:

        points = 0

        if features.car_loan == 1 or features.mortgage == 1:
            points = self._scores_list[3]
        else:
            points = self._scores_list[1]

        return points

    # Оценка по внешнему источнику 2 (ext_source_2)
    def _external2_score(self, features: Features) -> int:

        points = 0
        if features.ext_source_2 == 1:
            points = self._scores_list[3]
        else:
            points = self._scores_list[0]

        return points

    # Оценка по микрозаймам
    def _microloan_score(self, features: Features) -> int:

        points = 0

        if features.microloan == 0:
            points = self._scores_list[3]
        else:
            points = self._scores_list[0]

        return points
