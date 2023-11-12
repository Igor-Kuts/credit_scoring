from dataclasses import dataclass
from enum import Enum, auto


class ScoringDecision(Enum):
    """Возможные решения модели."""

    accepted = auto()
    declined = auto()


@dataclass
class ScoringResult(object):
    """Класс с результатами скоринга."""

    decision: ScoringDecision
    amount: int
    threshold: float
    proba: float


@dataclass
class LoanAmount(object):
    """Размеры займов по грейдам."""

    min_loan: int = 20_000
    mid_loan: int = 50_000
    high_loan: int = 100_000
    max_loan: int = 250_000

@dataclass
class Features(object):
    """Признаки для предсказания моделью."""
    
    ext_source_1: float = 0.0
    ext_source_2: float = 0.0
    ext_source_3: float = 0.0
    cnt_children: int = 0
    amt_income_total: float = 0.0
    amt_credit: float = 0.0
    amt_annuity: float = 0.0
    amt_goods_price: float = 0.0
    days_birth: int = 0
    days_employed: int = 0
    days_registration: int = 0
    days_id_publish: int = 0
    cnt_fam_members: float = 0.0
    region_rating_client: int = 0
    region_rating_client_w_city: int = 0
    hour_appr_process_start: int = 0
    years_beginexpluatation_avg: float = 0.0
    floorsmax_avg: float = 0.0
    livingarea_avg: float = 0.0
    years_beginexpluatation_mode: float = 0.0
    floorsmax_mode: float = 0.0
    livingarea_mode: float = 0.0
    years_beginexpluatation_medi: float = 0.0
    floorsmax_medi: float = 0.0
    livingarea_medi: float = 0.0
    totalarea_mode: float = 0.0
    obs_30_cnt_social_circle: float = 0.0
    def_30_cnt_social_circle: float = 0.0
    obs_60_cnt_social_circle: float = 0.0
    def_60_cnt_social_circle: float = 0.0
    days_last_phone_change: float = 0.0
    amt_req_credit_bureau_hour: float = 0.0
    amt_req_credit_bureau_day: float = 0.0
    amt_req_credit_bureau_week: float = 0.0
    amt_req_credit_bureau_mon: float = 0.0
    amt_req_credit_bureau_qrt: float = 0.0
    amt_req_credit_bureau_year: float = 0.0
    num_documents: int = 0
    building_info: int = 0
    age_years: int = 0
    id_change_age: int = 0
    id_age_change_diff: int = 0
    flag_late_id_change: int = 0
    annuity_to_income_rate: float = 0.0
    children_per_adult: float = 0.0
    inc_per_child: float = 0.0
    inc_per_adult: float = 0.0
    annuity_rate: float = 0.0
    weighted_ext_scores: float = 0.0
    composite_metric: float = 0.0
    cnt_late_payments: float = 0.0
    early_payments: float = 0.0
    avg_pct_instalment_paid: float = 0.0
    max_overdue_amt: float = 0.0
    min_overdue_amt: float = 0.0
    another_type_of_loan: float = 0.0
    car_loan: float = 0.0
    cash_loan__non_earmarked_: float = 0.0
    consumer_credit: float = 0.0
    credit_card: float = 0.0
    interbank_credit: float = 0.0
    loan_for_business_development: float = 0.0
    loan_for_purchase_of_shares__margin_lending_: float = 0.0
    loan_for_the_purchase_of_equipment: float = 0.0
    loan_for_working_capital_replenishment: float = 0.0
    microloan: float = 0.0
    mobile_operator_loan: float = 0.0
    mortgage: float = 0.0
    real_estate_loan: float = 0.0
    unknown_type_of_loan: float = 0.0
    cnt_overdue_another_type_of_loan: float = 0.0
    cnt_overdue_car_loan: float = 0.0
    cnt_overdue_cash_loan__non_earmarked_: float = 0.0
    cnt_overdue_consumer_credit: float = 0.0
    cnt_overdue_credit_card: float = 0.0
    cnt_overdue_loan_for_business_development: float = 0.0
    cnt_overdue_loan_for_the_purchase_of_equipment: float = 0.0
    cnt_overdue_loan_for_working_capital_replenishment: float = 0.0
    cnt_overdue_microloan: float = 0.0
    cnt_overdue_mortgage: float = 0.0
    cnt_overdue_real_estate_loan: float = 0.0
    cnt_overdue_unknown_type_of_loan: float = 0.0
    cnt_closed_another_type_of_loan: float = 0.0
    cnt_closed_car_loan: float = 0.0
    cnt_closed_cash_loan__non_earmarked_: float = 0.0
    cnt_closed_consumer_credit: float = 0.0
    cnt_closed_credit_card: float = 0.0
    cnt_closed_interbank_credit: float = 0.0
    cnt_closed_loan_for_business_development: float = 0.0
    cnt_closed_loan_for_purchase_of_shares__margin_lending_: float = 0.0
    cnt_closed_loan_for_the_purchase_of_equipment: float = 0.0
    cnt_closed_loan_for_working_capital_replenishment: float = 0.0
    cnt_closed_microloan: float = 0.0
    cnt_closed_mortgage: float = 0.0
    cnt_closed_real_estate_loan: float = 0.0
    cnt_closed_unknown_type_of_loan: float = 0.0
    cnt_overdue_1_dpd: float = 0.0
    cnt_overdue_2_dpd: float = 0.0
    cnt_overdue_3_dpd: float = 0.0
    cnt_overdue_4_dpd: float = 0.0
    cnt_overdue_5_dpd: float = 0.0
    days_since_last_actv_credit: float = 0.0
    num_prev_cash_loans: float = 0.0
    num_prev_consumer_loans: float = 0.0
    num_prev_revolving_loans: float = 0.0
    num_prev_xna: float = 0.0
    avg_pct_approved: float = 0.0
    avg_inq_freq: float = 0.0
    cnt_early_applications: float = 0.0
    cnt_daytime_applications: float = 0.0
    cnt_afterwork_applications: float = 0.0
    cnt_late_applications: float = 0.0
    name_contract_type: str = 'other'
    code_gender: str = 'other'
    flag_own_car: bool = 0
    flag_own_realty: bool = 0
    name_type_suite: str = 'other'
    name_income_type: str = 'other'
    name_education_type: str = 'other'
    name_family_status: str = 'other'
    name_housing_type: str = 'other'
    flag_emp_phone: bool = 0
    flag_work_phone: bool = 0
    flag_cont_mobile: bool = 0
    flag_phone: bool = 0
    flag_email: bool = 0
    occupation_type: str = 'other'
    reg_region_not_live_region: bool = 0
    reg_region_not_work_region: bool = 0
    live_region_not_work_region: bool = 0
    reg_city_not_live_city: bool = 0
    reg_city_not_work_city: bool = 0
    live_city_not_work_city: bool = 0
    organization_type: str = 'other'
    housetype_mode: str = 'other'
    emergencystate_mode: str = 'other'
    flag_document_2: bool = 0
    flag_document_3: bool = 0
    flag_document_5: bool = 0
    flag_document_6: bool = 0
    flag_document_7: bool = 0
    flag_document_8: bool = 0
    flag_document_9: bool = 0
    flag_document_11: bool = 0
    flag_document_13: bool = 0
    flag_document_14: bool = 0
    flag_document_15: bool = 0
    flag_document_16: bool = 0
    flag_document_17: bool = 0
    flag_document_18: bool = 0
    flag_document_19: bool = 0
    flag_document_20: bool = 0
    flag_document_21: bool = 0
