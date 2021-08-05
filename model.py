# %%

import seaborn as sns
from cadCAD import configs
import pandas as pd
from cadCAD.engine import ExecutionMode, ExecutionContext, Executor
from cadCAD.configuration.utils import config_sim
from cadCAD.configuration import Experiment
from cadCAD_tools.execution import easy_run
from typing import List, Dict
from cadCAD_tools.utils import generic_suf
from cadCAD_tools.preparation import prepare_params, prepare_state
from cadCAD_tools.types import InitialState, InitialValue, Param, ParamSweep, Signal, State, SystemParameters, TimestepBlock
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from random import random, normalvariate, sample
Fiat = float
Fiat_Per_Timestep = float
Timestep = int
Percentage = float
Percentage_Per_Timestep = float


@dataclass
class Employer:
    balance: Fiat
    consumer_utility: Fiat_Per_Timestep


@dataclass
class Employee:
    balance: Fiat


@dataclass
class Contract:
    employer: str
    wage: Fiat_Per_Timestep
    start: Timestep
    duration: Timestep

    @property
    def end(self) -> Timestep:
        return self.start + self.duration


N_employees = 15
N_employers = 3
N_contracts = 10

initial_employees = {key: Employee(normalvariate(12, 3))
                     for key in range(N_employees)}

initial_employers = {key: Employer(normalvariate(5200, 500),
                                   normalvariate(250, 50))
                     for key in range(N_employers)}


initial_contracts = {key: Contract(sample(initial_employers.keys(), 1)[0],
                                   normalvariate(2.5, 1),
                                   0,
                                   int(normalvariate(15, 3)))
                     for key in range(N_contracts)}


initial_state: InitialState = {
    'employers': InitialValue(initial_employers, Dict[str, Employer]),
    'employees': InitialValue(initial_employees, Dict[str, Employee]),
    'contracts': InitialValue(initial_contracts, Dict[str, Contract])
}

initial_state = prepare_state(initial_state)

params: SystemParameters = {
    # Hiring parameters
    'contract_duration': Param(20, Timestep),

    #
    'consumer_utility_decay': Param(0.05, Percentage_Per_Timestep),
    'mean_effort_to_utility': Param(1.5, Percentage),
    'std_effort_to_utility': Param(0.5, Percentage),

    # Critical wage related parameters
    'discount_rate': Param(1.0, Percentage_Per_Timestep),
    'unemployment_utility': ParamSweep([-10, 0.0, 10], Fiat),
    'shirk_detect_probability': ParamSweep([0.05, 0.5, 0.95], Percentage),
    'mean_effort': Param(2, Fiat_Per_Timestep),
    'std_effort': Param(0.2, Fiat_Per_Timestep)
}

params = prepare_params(params, cartesian_sweep=True)


def add_consumer_utility(employer: Employer,
                         decay: Percentage) -> Employer:
    return Employer(employer.balance * 0.95 + employer.consumer_utility,
                    employer.consumer_utility * (1 - decay))


def pay_employee(employee: Employee,
                 employer: Employer,
                 wage) -> tuple[Employee, Employer]:
    updated_employee = Employee(employee.balance + wage)
    updated_employer = Employer(
        employer.balance - wage, employer.consumer_utility)
    return (updated_employee, updated_employer)


def p_consumer_demand(params, _2, _3, state: State) -> Signal:
    decay = params['consumer_utility_decay']
    employers = {key: add_consumer_utility(employer, decay)
                 for key, employer
                 in state['employers'].items()}
    return {'employers': employers}


def p_wage_payments(_1, _2, _3, state) -> Signal:
    contracts = state['contracts']
    employees = state['employees'].copy()
    employers = state['employers'].copy()

    active_employees = {k: v
                        for k, v in employees.items()
                        if k in contracts}

    for key, employee in active_employees.items():
        contract = contracts[key]
        employer_key = contract.employer
        employer = employers[employer_key]
        wage = contract.wage

        updated_employee, update_employer = pay_employee(
            employee, employer, wage)
        employees[key] = updated_employee
        employers[employer_key] = update_employer

    return {'employees': employees,
            'employers': employers}


def critical_wage(effort: Fiat_Per_Timestep,
                  discount_rate: Percentage_Per_Timestep,
                  dismiss_probability: Percentage_Per_Timestep,
                  shirk_detect_probability: Percentage,
                  unemployment_utility: Fiat) -> float:

    value = discount_rate * unemployment_utility
    value += effort * (discount_rate
                       + dismiss_probability
                       + shirk_detect_probability) / shirk_detect_probability
    return value


def p_work(params, _2, _3, state) -> Signal:
    shirk_detect_probability = params['shirk_detect_probability']
    contracts: dict[str, Contract] = state['contracts'].copy()
    employees: dict[str, Employee] = state['employees'].copy()
    employers: dict[str, Employer] = state['employers'].copy()
    timestep = state['timestep']

    for (employee_key, employee) in employees.items():
        is_employed = (employee_key in contracts)
        if is_employed:
            contract: Contract = contracts[employee_key]
            employer: Employer = employers[contract.employer]

            # Compute random effort value
            effort = normalvariate(mu=params['mean_effort'],
                                   sigma=params['std_effort'])

            # Compute dismissal probability based on contract length
            contract_length = contract.duration
            remaining_contract_length = contract.end - timestep
            remaining_contract_fraction = remaining_contract_length / contract_length
            dismiss_probability = 1 - remaining_contract_fraction

            # Compute critical wage
            employee_critical_wage = critical_wage(effort,
                                                   params['discount_rate'],
                                                   dismiss_probability,
                                                   shirk_detect_probability,
                                                   params['unemployment_utility'])

            capped_relative_wage = max(employee_critical_wage, contract.wage)

            # Compute shirk probability if it is less than the critical wage
            shirk_probability = 1 - capped_relative_wage
            shirk = random() < shirk_probability

            if shirk:
                was_detected = random() < shirk_detect_probability
                if was_detected:
                    # Finish contract
                    contracts.pop(employee_key)
                else:
                    # Add utility to the employee
                    employee.balance += effort
                    pass
            else:
                # Add utility to employer
                utility_factor = normalvariate(params['mean_effort_to_utility'],
                                               params['std_effort_to_utility'])
                new_utility = effort * utility_factor
                employer.consumer_utility += new_utility
                pass
        else:
            continue

    return {'employees': employees,
            'contracts': contracts,
            'employers': employers}


def p_fire(params, _2, _3, state) -> Signal:
    contracts: dict[str, Contract] = state['contracts'].copy()
    timestep = state['timestep']

    contracts = {key: contract for key, contract in contracts.items()
                 if contract.end > timestep}

    return {'contracts': contracts}


def p_hire(params, _2, _3, state) -> Signal:

    duration = params['contract_duration']
    contracts: dict[str, Contract] = state['contracts'].copy()
    employee_pool: dict[str, Employee] = state['employees'].copy()
    employers: dict[str, Employer] = state['employers']
    timestep = state['timestep']

    unemployed: dict[str, Employee] = {employee_key: employee
                                       for employee_key, employee
                                       in employee_pool.items()
                                       if employee_key not in contracts}

    for candidate_key, candidate in unemployed.items():
        # TODO

        # Compute candidate wage
        expected_wage = np.percentile([contract.wage
                                       for contract
                                       in contracts.values()], 90)
        if np.isnan(expected_wage):
            expected_wage = 0.0
        else:
            pass

        survival_wage = candidate.balance / 12  # TODO parametrize
        wage_proposal = 1.5 * \
            (expected_wage + survival_wage) / 2  # TODO parametrize

        # Compute critical wage
        expected_effort = params['mean_effort']
        expected_discount_rate = params['discount_rate']
        expected_dismiss_probability = 1 / duration

        effective_unemployment_utility = params['unemployment_utility'] + \
            expected_wage + survival_wage

        wage_critical = critical_wage(expected_effort,
                                      expected_discount_rate,
                                      expected_dismiss_probability,
                                      params['shirk_detect_probability'],
                                      effective_unemployment_utility)

        # Compute expensive wage threshold
        expected_utility = expected_effort * params['mean_effort_to_utility']
        expected_utility -= wage_proposal
        expected_utility *= duration

        if timestep > 25:
            pass
        else:
            pass

        if expected_utility > 0:
            if wage_proposal > wage_critical:
                contracts[candidate_key] = Contract(sample(employers.keys(), 1)[0],
                                                    wage_proposal,
                                                    timestep,
                                                    duration)
                pass
            else:
                # Hire with probability
                hire_probability = 0.5  # TODO parametrize
                if hire_probability > random():
                    contracts[candidate_key] = Contract(sample(employers.keys(), 1)[0],
                                                        wage_proposal,
                                                        timestep,
                                                        duration)
                else:
                    continue
        else:
            # Hire with probability
            hire_probability = 0.05  # TODO parametrize
            if hire_probability > random():
                contracts[candidate_key] = Contract(sample(employers.keys(), 1)[0],
                                                    wage_proposal,
                                                    timestep,
                                                    duration)
            else:
                continue

    return {'contracts': contracts}


def pay_living_costs(employee: Employee) -> Employee:
    # TODO parametrize
    new_balance = employee.balance * 0.8
    return Employee(new_balance)


def p_living_costs(params, _2, _3, state):
    employees = state['employees'].copy()

    employees = {key: pay_living_costs(employee)
                 for key, employee
                 in employees.items()}

    return {'employees': employees}


timestep_block: TimestepBlock = [
    {
        'label': 'Consumer demand / Wage payments / Employee living costs',
        'policies': {
            'consumer_demand': p_consumer_demand,

        },
        'variables': {
            'employers': generic_suf('employers')

        }
    },
    {
        'label': 'Consumer demand / Wage payments / Employee living costs',
        'policies': {
            'living_costs': p_living_costs

        },
        'variables': {
            'employees': generic_suf('employees')

        }
    },
    {
        'label': 'Consumer demand / Wage payments / Employee living costs',
        'policies': {
            'wage_payments': p_wage_payments,

        },
        'variables': {
            'employers': generic_suf('employers'),
            'employees': generic_suf('employees')

        }
    },
    {
        'label': 'Hire Employees',
        'policies': {
            'hire': p_hire
        },
        'variables': {
            'contracts': generic_suf('contracts')
        }
    },
    {
        'label': 'Fire Employees',
        'policies': {
            'fire_by_duration': p_fire
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
            'employees': generic_suf('employees'),
            'contracts': generic_suf('contracts')
        }
    },
]

# %%

df = easy_run(initial_state,
              params,
              timestep_block,
              200,
              10)

# %%


def f(x): return pd.DataFrame.from_dict(x, orient='index')


s = df.employers.apply(f)
employers_df = pd.concat(s.values)
employers_df.head(10)

# %%


def flatten_dict(key: str,
                 value: dict[object, dict]) -> dict[str, object]:
    output_dict = {}
    for inner_index, inner_dataclass in value.items():
        for k, v in inner_dataclass.__dict__.items():
            output_dict[f"{key}_{inner_index}_{k}"] = v
    return output_dict


records = df.to_dict(orient='records')
# %%
flatten_dict('employers', records[0]['employers'])
# %%
clean_records = []
for record in records:
    clean_record: dict[str, object] = {}
    for key, value in record.items():
        if type(value) is dict:
            flattened = flatten_dict(key, value)
            clean_record = clean_record | flattened
        else:
            clean_record[key] = value
    clean_records.append(clean_record)

# %%
df = pd.DataFrame(clean_records)

employee_cols = set(col
                    for col in df.columns
                    if 'employee' in col)
employer_balance_cols = set(col
                            for col in df.columns
                            if 'employer' in col
                            and 'balance' in col)
employer_utility_cols = set(col
                            for col in df.columns
                            if 'employer' in col
                            and 'utility' in col)


wage_cols = set(col
                for col in df.columns
                if 'contract' in col
                and 'wage' in col)
# %%
df = (df.assign(mean_employee_balance=df[employee_cols].mean(axis=1))
      .assign(mean_employer_balance=df[employer_balance_cols].mean(axis=1))
      .assign(mean_employer_utility=df[employer_utility_cols].mean(axis=1))
      .assign(mean_wage=df[wage_cols].mean(axis=1))
      )
# %%
plt.figure(figsize=(10, 5))
sns.lineplot(data=df,
             x='timestep',
             y='mean_employee_balance',
             hue='unemployment_utility',
             style='shirk_detect_probability')
plt.show()
# %%
plt.figure(figsize=(10, 5))
sns.lineplot(data=df,
             x='timestep',
             y='mean_employer_balance',
             hue='unemployment_utility',
             style='shirk_detect_probability')
plt.show()
# %%
plt.figure(figsize=(10, 5))
sns.lineplot(data=df,
             x='timestep',
             y='mean_employer_utility',
             hue='unemployment_utility',
             style='shirk_detect_probability')
plt.show()
# %%
plt.figure(figsize=(10, 5))
sns.lineplot(data=df,
             x='timestep',
             y='mean_wage',
             hue='unemployment_utility',

             style='shirk_detect_probability')
plt.show()
# %%
