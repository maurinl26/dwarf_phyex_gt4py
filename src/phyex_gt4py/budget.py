from dataclasses import dataclass, field


@dataclass
class TBudgetConf:
    bu_enable: bool = field(default=False)
    budget_u: bool = field(default=False)
    budget_v: bool = field(default=False)
    budget_w: bool = field(default=False)
    budget_th: bool = field(default=False)
    budget_tke: bool = field(default=False)
    budget_rv: bool = field(default=False)
    budget_rc: bool = field(default=False)
    budget_rr: bool = field(default=False)
    budget_ri: bool = field(default=False)
    budget_rs: bool = field(default=False)
    budget_rg: bool = field(default=False)
    budget_rh: bool = field(default=False)
    budget_sv: bool = field(default=False)
