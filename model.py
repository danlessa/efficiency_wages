from typing import List, Dict
from cadCAD_tools.utils import generic_suf
from cadCAD_tools.types import InitialState, InitialValue, Parameter, Signal, State, SystemParameters, TimestepBlock
from dataclasses import dataclass

from random import random
Fiat = float
Fiat_Per_Timestep = float
Timestep = int


@dataclass
class Employer:
    label: str
    balance: Fiat
    consumer_utility: Fiat_Per_Timestep


@dataclass
class Employee:
    label: str
    balance: Fiat


@dataclass
class Contract:
    employer: str
    wage: Fiat_Per_Timestep
    duration: Timestep


initial_state: InitialState = {
    'employers': InitialValue(None, List[Employer]),
    'employees': InitialValue(None, List[Employee]),
    'contracts': InitialValue(None, Dict[str, Contract]),
}

params: SystemParameters = {
    'shirk_detection_probability': Parameter(0.1, float)
}


def add_consumer_utility(employer: Employer) -> Employer:
    return Employer(employer.label,
                    employer.balance + employer.consumer_utility,
                    employer.consumer_utility)


def pay_employee(employee: Employee,
                 contracts: Dict[str, Contract]) -> Employee:
    if employee.label in contracts:
        return Employee(employee.label,
                        employee.balance + contracts[employee.label].wage)
    else:
        return employee


def p_consumer_demand(_1, _2, _3, state: State) -> Signal:
    employers = [add_consumer_utility(employer)
                 for employer
                 in state['employers']]
    return {'employers': employers}


def p_wage_payments(_1, _2, _3, state) -> Signal:
    employees = [pay_employee(employee, state['contracts'])
                 for employee in state['employees']]
    return {'employees': employees}


def critical_wage(effort: float,
                  discount_rate: float,
                  dismissal_probability_per_unit_of_time: float,
                  shirk_detection_probability: float,
                  unemployment_utility: float=0.0) -> float:

    value = discount_rate * unemployment_utility
    value += effort * (discount_rate
                       + dismissal_probability_per_unit_of_time
                       + shirk_detection_probability) / shirk_detection_probability
    return value


def p_work(_1, _2, _3, state) -> Signal:
    contracts = state['contracts']
    employees = state['employees']

    for employee in employees:
        is_employed = (employee in contracts)
        if is_employed:
            employee_critical_wage = critical_wage(effort,
            discount_rate,
            dis )
            wage = contracts[employee.label].wage
            capped_relative_wage = max(employee_critical_wage, wage)
            shirk_probability = 1 - capped_relative_wage
            shirk = random() < shirk_probability
            if shirk:
                pass
            else:
                pass
        else:
            continue
    pass


timestep_block: TimestepBlock = [
    {
        'label': 'Consumer demand / Wage payments',
        'policies': {
            'consumer_demand': p_consumer_demand,
            'wage_payments': p_wage_payments

        },
        'variables': {
            'employers': generic_suf('employers'),
            'employees': generic_suf('employees')

        }
    },
    {
        'label': 'Hire / Fire Employees',
        'policies': {
            'fire_by_duration': None,
            'fire_by_shirk': None,
            'hire': None
        },
        'variables': {
            'contracts': generic_suf('contracts')
        }
    },
    {
        'label': 'Shirk / Work',
        'policies': {
            'work_or_shirk': p_work
        },
        'variables': {
            'employers': generic_suf('employers'),
            'employees': generic_suf('employees')
        }
    },
]
