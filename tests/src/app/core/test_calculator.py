import os
import sys
from itertools import product

import pytest

sys.path.insert(1, os.path.join(sys.path[0], '../../../..'))

from src.app.core.api import Features
from src.app.core.calculator import Calculator


class TestCalculator(object):
    """Юнит-тесты для методов калькулятора."""

    # Значения для тестирования подсчёта финального скора
    age_scores = [0, 3, 5]
    emp_days_scores = [0, 1, 3, 5]
    income_scores = [0, 1, 3, 5]
    credit_type_scores = [1, 5]
    external2_scores = [0, 5]
    microloan_scores = [0, 5]
    
    values = list(product(
        age_scores,
        emp_days_scores,
        income_scores,
        credit_type_scores,
        external2_scores,
        microloan_scores,
    ))
    values_list = [x+(sum(x),) for x in values]

    @pytest.mark.parametrize(
        ('proba', 'final_score', 'correct_result'),
        [
            (0.07, 15, 250_000),
            (0.11, 32, 250_000),
            (0.13, 27, 100_000),
            (0.10, 15, 50_000),
            (0.07, 15, 250_000),
            (0.085, 5, 20_000),       
        ],
    )
    def test_calc_amount(self, proba, final_score, correct_result):
        """Тестирование метода calc_amount"""
        calculator = Calculator()
        assert calculator.calc_amount(proba, final_score) == correct_result
    
    @pytest.mark.parametrize(
        ('age_scores', 'emp_days_scores', 'income_scores', 
         'credit_type_scores', 'external2_scores', 'microloan_scores', 'correct_result'),
        values_list
    )
    def test_score_for_sum(self,
                           age_scores,
                           emp_days_scores,
                           income_scores,
                           credit_type_scores,
                           external2_scores,
                           microloan_scores,
                           correct_result):
        """Тестирование метода score_for_sum"""
        scores = [age_scores, emp_days_scores, income_scores, credit_type_scores,
                      external2_scores, microloan_scores]
        assert sum(scores) == correct_result

    @pytest.mark.parametrize(
        ('age_years', 'correct_result'),
        [
            (15, 0),
            (70, 0),
            (35, 5),
            (55, 3),
        ],
    )
    def test_age_score(self, age_years, correct_result):
        """Тестирование метода _age_score"""
        calculator = Calculator()
        test_case = Features(age_years=age_years)
        assert calculator._age_score(test_case) == correct_result

    @pytest.mark.parametrize(
        ('days_employed', 'correct_result'),
        [
            (150, 0),
            (750, 1),
            (1500, 3),
            (1826, 5),
        ],
    )
    def test_emp_days_score(self, days_employed, correct_result):
        """Тестирование метода _emp_days_score"""
        calculator = Calculator()
        test_case = Features(days_employed=days_employed)
        assert calculator._emp_days_score(test_case) == correct_result

    @pytest.mark.parametrize(
        ('amt_income_total', 'correct_result'),
        [
            (10_000, 0),
            (50_000, 0),
            (100_000, 1),
            (200_000, 3),
            (300_000, 5),
        ],
    )
    def test_income_score(self, amt_income_total, correct_result):
        """Тестирование метода _income_score"""
        calculator = Calculator()
        test_case = Features(amt_income_total=amt_income_total)
        assert calculator._income_score(test_case) == correct_result

    @pytest.mark.parametrize(
        ('car_loan', 'mortgage', 'correct_result'),
        [
            (1, 1, 5),
            (1, 0, 5),
            (0, 1, 5),
            (0, 0, 1),
        ],
    )
    def test_credit_type_score(self, car_loan, mortgage, correct_result):
        """Тестирование метода _credit_type_score"""
        calculator = Calculator()
        test_case = Features(
            car_loan=car_loan,
            mortgage=mortgage,
        )
        assert calculator._credit_type_score(test_case) == correct_result

    @pytest.mark.parametrize(
        ('ext_source_2', 'correct_result'),
        [
            (1, 5),
            (0, 0),
        ],
    )
    def test_external2_score(self, ext_source_2, correct_result):
        """Тестирование метода _eternal2_score"""
        calculator = Calculator()
        test_case = Features(ext_source_2=ext_source_2)
        assert calculator._external2_score(test_case) == correct_result

    @pytest.mark.parametrize(
        ('microloan', 'correct_result'),
        [
            (0, 5),
            (3, 0),
        ],
    )
    def test_microloan_score(self, microloan, correct_result):
        """Тестирование метода _microloan_score"""
        calculator = Calculator()
        test_case = Features(microloan=microloan)
        assert calculator._microloan_score(test_case) == correct_result
