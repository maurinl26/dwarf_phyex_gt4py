from dataclasses import dataclass, field

@dataclass
class TBudgetConf:
    bu_enable = field(default=False)
    budget_u: field(default=False)
    budget_v = field(default=False)
    budget_w = field(default=False)
    budget_th = field(default=False)
    budget_tke = field(default=False)
    budget_rv = field(defautl=False)
    budget_rc = field(default=False)
    budget_rr = field(default=False)
    budget_ri = field(default=False)
    budget_rs = field(default=False)
    budget_rg = field(default=False)
    budget_rh = field(default=False)
    budget_sv = field(default=False)