"""
MarxTown MPI — A software-style system to measure a town’s economic productivity using Marxist ideas.

Single-file FastAPI app with:
- SQLite DB via SQLAlchemy (easily swap to Postgres)
- Data models & ingestion endpoints
- Metrics pipeline implementing:
    * Surplus Value (s), Variable Capital (v), Constant Capital (c proxy)
    * Rate of Exploitation e = s / v
    * Profit Rate r = s / (c + v)
    * Reinvestment Ratio (local) = local_capex / s
    * Living Wage Gap = (avg_wage - living_wage) / living_wage
    * Labor Utilization = employed_hours / potential_hours
    * External Extraction Ratio = net_outflows / s
- Composite Marxist Productivity Index (MPI) and sub-indices

Run:
  1) `pip install fastapi uvicorn sqlalchemy pydantic pandas python-dateutil`
  2) `uvicorn app:app --reload`

Notes:
- Metrics are period-based (monthly/quarterly/annual). Provide `period_start`, `period_end`.
- If you lack some inputs, the system falls back to proxies (documented below).
"""
from __future__ import annotations

from datetime import date, datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Body, Query
from pydantic import BaseModel, Field

from sqlalchemy import (
    create_engine, Column, Integer, Float, String, Date, ForeignKey, Boolean, func
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
import pandas as pd
from dateutil.relativedelta import relativedelta

# ------------------------
# DB setup
# ------------------------
DATABASE_URL = "sqlite:///./marxtown.db"  # swap to Postgres by changing URL
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ------------------------
# Models
# ------------------------
class Sector(Base):
    __tablename__ = "sectors"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    # Marxist classification of labor in this sector (default); can be overridden per record
    labor_class_default = Column(String, default="productive")  # productive/unproductive/reproductive

class Firm(Base):
    __tablename__ = "firms"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    sector_id = Column(Integer, ForeignKey("sectors.id"), nullable=False)
    outside_owner_share = Column(Float, default=0.0)  # 0..1 share owned by external capital
    sector = relationship("Sector")

class EmploymentRecord(Base):
    __tablename__ = "employment_records"
    id = Column(Integer, primary_key=True)
    firm_id = Column(Integer, ForeignKey("firms.id"), nullable=False)
    person_id = Column(String, nullable=True)  # optional anon id
    period_start = Column(Date, nullable=False)
    period_end = Column(Date, nullable=False)
    hours = Column(Float, nullable=False)
    wage_income = Column(Float, nullable=False)  # total wages paid for the period
    labor_class = Column(String, nullable=True)  # overrides sector default if present
    firm = relationship("Firm")

class FirmAccounts(Base):
    __tablename__ = "firm_accounts"
    id = Column(Integer, primary_key=True)
    firm_id = Column(Integer, ForeignKey("firms.id"), nullable=False)
    period_start = Column(Date, nullable=False)
    period_end = Column(Date, nullable=False)
    revenue = Column(Float, default=0.0)  # sales
    intermediate_inputs = Column(Float, default=0.0)  # cost of goods/services bought in
    depreciation = Column(Float, default=0.0)  # proxy for constant capital consumed
    capital_expenditure_local = Column(Float, default=0.0)  # reinvested locally this period
    profits_distributed = Column(Float, default=0.0)
    interest_rent_paid_outside = Column(Float, default=0.0)  # net to outside
    taxes_local = Column(Float, default=0.0)
    taxes_outside = Column(Float, default=0.0)
    firm = relationship("Firm")

class HouseholdCostBasket(Base):
    __tablename__ = "household_cost_basket"
    id = Column(Integer, primary_key=True)
    period_start = Column(Date, nullable=False)
    period_end = Column(Date, nullable=False)
    # living wage proxy: monthly/period cost of reproducing labor power per full-time worker
    housing = Column(Float, default=0.0)
    food = Column(Float, default=0.0)
    healthcare = Column(Float, default=0.0)
    childcare = Column(Float, default=0.0)
    transport = Column(Float, default=0.0)
    education = Column(Float, default=0.0)
    other = Column(Float, default=0.0)

class Demography(Base):
    __tablename__ = "demography"
    id = Column(Integer, primary_key=True)
    period_start = Column(Date, nullable=False)
    period_end = Column(Date, nullable=False)
    working_age_population = Column(Integer, nullable=False)  # 15–64 or as defined
    participation_rate = Column(Float, default=0.0)  # 0..1
    standard_week_hours = Column(Float, default=40.0)

Base.metadata.create_all(bind=engine)

# ------------------------
# Schemas
# ------------------------
class SectorIn(BaseModel):
    name: str
    labor_class_default: str = Field("productive", regex="^(productive|unproductive|reproductive)$")

class FirmIn(BaseModel):
    name: str
    sector_id: int
    outside_owner_share: float = Field(0.0, ge=0, le=1)

class EmploymentIn(BaseModel):
    firm_id: int
    person_id: Optional[str] = None
    period_start: date
    period_end: date
    hours: float
    wage_income: float
    labor_class: Optional[str] = Field(None, regex="^(productive|unproductive|reproductive)$")

class AccountsIn(BaseModel):
    firm_id: int
    period_start: date
    period_end: date
    revenue: float = 0.0
    intermediate_inputs: float = 0.0
    depreciation: float = 0.0
    capital_expenditure_local: float = 0.0
    profits_distributed: float = 0.0
    interest_rent_paid_outside: float = 0.0
    taxes_local: float = 0.0
    taxes_outside: float = 0.0

class BasketIn(BaseModel):
    period_start: date
    period_end: date
    housing: float = 0.0
    food: float = 0.0
    healthcare: float = 0.0
    childcare: float = 0.0
    transport: float = 0.0
    education: float = 0.0
    other: float = 0.0

class DemoIn(BaseModel):
    period_start: date
    period_end: date
    working_age_population: int
    participation_rate: float = Field(0.0, ge=0, le=1)
    standard_week_hours: float = 40.0

# ------------------------
# FastAPI app
# ------------------------
app = FastAPI(title="MarxTown MPI", version="0.1.0")

# Utility

def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ------------------------
# Ingestion endpoints (minimal)
# ------------------------
@app.post("/sectors")
def create_sector(payload: SectorIn):
    db = SessionLocal()
    try:
        s = Sector(name=payload.name, labor_class_default=payload.labor_class_default)
        db.add(s)
        db.commit()
        db.refresh(s)
        return {"id": s.id}
    finally:
        db.close()

@app.post("/firms")
def create_firm(payload: FirmIn):
    db = SessionLocal()
    try:
        f = Firm(name=payload.name, sector_id=payload.sector_id, outside_owner_share=payload.outside_owner_share)
        db.add(f)
        db.commit()
        db.refresh(f)
        return {"id": f.id}
    finally:
        db.close()

@app.post("/employment")
def add_employment(payload: EmploymentIn):
    db = SessionLocal()
    try:
        rec = EmploymentRecord(**payload.dict())
        db.add(rec)
        db.commit()
        db.refresh(rec)
        return {"id": rec.id}
    finally:
        db.close()

@app.post("/accounts")
def add_accounts(payload: AccountsIn):
    db = SessionLocal()
    try:
        acc = FirmAccounts(**payload.dict())
        db.add(acc)
        db.commit()
        db.refresh(acc)
        return {"id": acc.id}
    finally:
        db.close()

@app.post("/basket")
def add_basket(payload: BasketIn):
    db = SessionLocal()
    try:
        b = HouseholdCostBasket(**payload.dict())
        db.add(b)
        db.commit()
        db.refresh(b)
        return {"id": b.id}
    finally:
        db.close()

@app.post("/demography")
def add_demo(payload: DemoIn):
    db = SessionLocal()
    try:
        d = Demography(**payload.dict())
        db.add(d)
        db.commit()
        db.refresh(d)
        return {"id": d.id}
    finally:
        db.close()

# ------------------------
# Metrics computation helpers
# ------------------------

def _period_filter(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    mask = (df["period_start"] >= pd.Timestamp(start)) & (df["period_end"] <= pd.Timestamp(end))
    return df.loc[mask].copy()


def load_frames(db, start: date, end: date) -> Dict[str, pd.DataFrame]:
    # Pull tables to DataFrames
    sectors = pd.read_sql(db.query(Sector).statement, db.bind)
    firms = pd.read_sql(db.query(Firm).statement, db.bind)
    emp = pd.read_sql(db.query(EmploymentRecord).statement, db.bind)
    acc = pd.read_sql(db.query(FirmAccounts).statement, db.bind)
    bask = pd.read_sql(db.query(HouseholdCostBasket).statement, db.bind)
    demo = pd.read_sql(db.query(Demography).statement, db.bind)

    # Convert dates
    for dfx in [emp, acc, bask, demo]:
        if not dfx.empty:
            dfx["period_start"] = pd.to_datetime(dfx["period_start"]).dt.date
            dfx["period_end"] = pd.to_datetime(dfx["period_end"]).dt.date

    # Filter
    emp = _period_filter(emp, start, end) if not emp.empty else emp
    acc = _period_filter(acc, start, end) if not acc.empty else acc
    bask = _period_filter(bask, start, end) if not bask.empty else bask
    demo = _period_filter(demo, start, end) if not demo.empty else demo

    return {"sectors": sectors, "firms": firms, "emp": emp, "acc": acc, "bask": bask, "demo": demo}


def compute_core_values(frames: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    sectors, firms, emp, acc, bask, demo = (
        frames["sectors"], frames["firms"], frames["emp"], frames["acc"], frames["bask"], frames["demo"]
    )

    # Merge employment with firms and sectors to classify labor
    if not emp.empty:
        emp = emp.merge(firms, left_on="firm_id", right_on="id", suffixes=("", "_firm"))
        emp = emp.merge(sectors, left_on="sector_id", right_on="id", suffixes=("", "_sector"))
        emp["labor_class_final"] = emp["labor_class"].fillna(emp["labor_class_default"])  # override default if present
    else:
        emp = pd.DataFrame(columns=["hours", "wage_income", "labor_class_final"])  # empty

    # Labor inputs
    total_hours = float(emp["hours"].sum()) if not emp.empty else 0.0
    productive_hours = float(emp.loc[emp["labor_class_final"] == "productive", "hours"].sum()) if not emp.empty else 0.0

    # Wages
    wages_total = float(emp["wage_income"].sum()) if not emp.empty else 0.0

    # Accounts:
    value_added = 0.0
    dep = 0.0
    capex_local = 0.0
    profits_dist = 0.0
    outflows = 0.0

    if not acc.empty:
        # Value Added proxy = Revenue - Intermediate Inputs
        value_added = float((acc["revenue"] - acc["intermediate_inputs"]).sum())
        dep = float(acc["depreciation"].sum())
        capex_local = float(acc["capital_expenditure_local"].sum())
        profits_dist = float(acc["profits_distributed"].sum())
        outflows = float((acc["interest_rent_paid_outside"] + acc["taxes_outside"]).sum())

    # Surplus value s = Value Added - Wages - Depreciation (proxy for c consumed)
    surplus_value = max(value_added - wages_total - dep, 0.0)

    # Constant capital c proxy: depreciation (consumed) + average working capital tied up in inputs (approx by intermediate_inputs)
    c_proxy = float(acc["intermediate_inputs"].sum()) + dep if not acc.empty else dep

    v = wages_total

    # Demography for potential hours
    if not demo.empty:
        # Assume period length in weeks from first row
        d0 = demo.iloc[0]
        period_weeks = max(1, int((pd.Timestamp(d0["period_end"]) - pd.Timestamp(d0["period_start"])) / pd.Timedelta(weeks=1)))
        potential_hours = float(demo["working_age_population"].mean() * demo["participation_rate"].mean() * d0["standard_week_hours"] * period_weeks)
    else:
        potential_hours = total_hours  # fallback

    # Basket: living wage per FTE
    if not bask.empty:
        basket_cost = float((bask[["housing","food","healthcare","childcare","transport","education","other"]].sum(axis=1)).mean())
    else:
        basket_cost = 0.0

    # Avg wage per FTE hour
    avg_wage_per_hour = (wages_total / total_hours) if total_hours > 0 else 0.0
    # Living wage per hour: assume basket is for a full-time worker per period; derive hourly by dividing by standard hours
    if not demo.empty and total_hours > 0:
        std_week = float(d0["standard_week_hours"]) if not demo.empty else 40.0
        # approximate FTE hours in period using employed headcount proxy via total_hours / std_week / period_weeks
        if not demo.empty:
            period_weeks = max(1, int((pd.Timestamp(d0["period_end"]) - pd.Timestamp(d0["period_start"])) / pd.Timedelta(weeks=1)))
        fte_hours = std_week * period_weeks
        living_wage_per_hour = (basket_cost / fte_hours) if fte_hours > 0 else 0.0
    else:
        living_wage_per_hour = 0.0

    return {
        "total_hours": total_hours,
        "productive_hours": productive_hours,
        "wages_total": wages_total,
        "value_added": value_added,
        "depreciation": dep,
        "surplus_value": surplus_value,
        "c_proxy": c_proxy,
        "v": v,
        "capex_local": capex_local,
        "profits_distributed": profits_dist,
        "net_outflows_external": outflows,
        "potential_hours": potential_hours,
        "avg_wage_per_hour": avg_wage_per_hour,
        "living_wage_per_hour": living_wage_per_hour,
    }


def compute_indices(core: Dict[str, float]) -> Dict[str, Any]:
    s = core["surplus_value"]
    v = core["v"]
    c = core["c_proxy"]
    total_hours = core["total_hours"]
    pot_hours = core["potential_hours"]

    # Base ratios
    exploitation_rate = (s / v) if v > 0 else 0.0  # e = s/v
    profit_rate = (s / (c + v)) if (c + v) > 0 else 0.0
    reinvestment_ratio = (core["capex_local"] / s) if s > 0 else 0.0  # local reinvestment of surplus
    labor_utilization = (total_hours / pot_hours) if pot_hours > 0 else 0.0
    external_extraction_ratio = (core["net_outflows_external"] / s) if s > 0 else 0.0

    # Living wage gap (negative is bad)
    lw = core["living_wage_per_hour"]
    aw = core["avg_wage_per_hour"]
    living_wage_gap = ((aw - lw) / lw) if lw > 0 else 0.0

    # --- Normalizations (0..1) ---
    # Value Creation Index (VCI): emphasize surplus produced per hour and utilization
    surplus_per_hour = (s / total_hours) if total_hours > 0 else 0.0
    # Normalize via logistic-like squash with scale anchors
    def squash(x, k=1.0, x0=0.0):
        import math
        return 1.0 / (1.0 + math.exp(-k * (x - x0)))

    VCI = 0.6 * squash(surplus_per_hour, k=5, x0=0.0) + 0.4 * min(max(labor_utilization, 0.0), 1.0)

    # Exploitation Index (XI): higher e and worse living wage gap => higher exploitation
    # Normalize e using squash around e=1 (100% exploitation)
    XI_raw = 0.7 * squash(exploitation_rate, k=1.2, x0=1.0) + 0.3 * squash(-living_wage_gap, k=2.0, x0=0.0)
    XI = min(max(XI_raw, 0.0), 1.0)

    # Reinvestment & Leakage Index (RLI): more local reinvestment and less external leakage -> higher
    # External leakage penalty
    leakage_penalty = min(max(external_extraction_ratio, 0.0), 2.0) / 2.0  # cap at 2
    RLI = min(max(0.7 * min(reinvestment_ratio, 1.5) / 1.5 + 0.3 * (1 - leakage_penalty), 0.0), 1.0)

    # Composite Marxist Productivity Index (MPI)
    # A town is "productive" in a Marxist sense when it creates surplus (VCI), 
    # but we reward systems that do so with minimal exploitation (1 - XI) and strong reinvestment (RLI).
    MPI = 0.4 * VCI + 0.3 * (1 - XI) + 0.3 * RLI

    return {
        "exploitation_rate": exploitation_rate,
        "profit_rate": profit_rate,
        "reinvestment_ratio": reinvestment_ratio,
        "labor_utilization": labor_utilization,
        "external_extraction_ratio": external_extraction_ratio,
        "living_wage_gap": living_wage_gap,
        "surplus_per_hour": surplus_per_hour,
        "VCI": VCI,
        "XI": XI,
        "RLI": RLI,
        "MPI": MPI,
    }

# ------------------------
# API: Metrics
# ------------------------
class MetricsOut(BaseModel):
    core: Dict[str, float]
    indices: Dict[str, float]


def _parse_dates(ps: Optional[date], pe: Optional[date]) -> (date, date):
    if ps and pe:
        return ps, pe
    # Default to last full month
    today = date.today().replace(day=1)
    end = today - relativedelta(days=1)
    start = end.replace(day=1)
    return start, end


@app.get("/metrics", response_model=MetricsOut)
def get_metrics(
    period_start: Optional[date] = Query(None),
    period_end: Optional[date] = Query(None),
):
    start, end = _parse_dates(period_start, period_end)
    db = SessionLocal()
    try:
        frames = load_frames(db, start, end)
        core = compute_core_values(frames)
        indices = compute_indices(core)
        return {"core": core, "indices": indices}
    finally:
        db.close()

# ------------------------
# Seed endpoint (optional demo data)
# ------------------------
@app.post("/seed-demo")
def seed_demo():
    db = SessionLocal()
    try:
        # basic sectors
        manuf = Sector(name="Manufacturing", labor_class_default="productive")
        health = Sector(name="Healthcare", labor_class_default="reproductive")
        retail = Sector(name="Retail", labor_class_default="unproductive")
        db.add_all([manuf, health, retail])
        db.commit()
        db.refresh(manuf); db.refresh(health); db.refresh(retail)

        # firms
        f1 = Firm(name="Town Gears Co.", sector_id=manuf.id, outside_owner_share=0.2)
        f2 = Firm(name="CareWell Clinic", sector_id=health.id, outside_owner_share=0.0)
        f3 = Firm(name="Main St. Mart", sector_id=retail.id, outside_owner_share=0.8)
        db.add_all([f1, f2, f3]); db.commit(); db.refresh(f1); db.refresh(f2); db.refresh(f3)

        # period (last month)
        today = date.today().replace(day=1)
        pend = today - relativedelta(days=1)
        pstart = pend.replace(day=1)

        # employment (hours & wages)
        emps = [
            EmploymentRecord(firm_id=f1.id, person_id="w1", period_start=pstart, period_end=pend, hours=160, wage_income=4800),
            EmploymentRecord(firm_id=f1.id, person_id="w2", period_start=pstart, period_end=pend, hours=160, wage_income=4400),
            EmploymentRecord(firm_id=f2.id, person_id="w3", period_start=pstart, period_end=pend, hours=160, wage_income=5200),
            EmploymentRecord(firm_id=f3.id, person_id="w4", period_start=pstart, period_end=pend, hours=120, wage_income=1920),
        ]
        db.add_all(emps)

        # accounts
        accs = [
            FirmAccounts(firm_id=f1.id, period_start=pstart, period_end=pend, revenue=120000, intermediate_inputs=60000, depreciation=8000, capital_expenditure_local=6000, profits_distributed=4000, interest_rent_paid_outside=2000, taxes_local=3000, taxes_outside=1000),
            FirmAccounts(firm_id=f2.id, period_start=pstart, period_end=pend, revenue=80000, intermediate_inputs=30000, depreciation=5000, capital_expenditure_local=2000, profits_distributed=1000, interest_rent_paid_outside=500, taxes_local=1500, taxes_outside=500),
            FirmAccounts(firm_id=f3.id, period_start=pstart, period_end=pend, revenue=50000, intermediate_inputs=25000, depreciation=2000, capital_expenditure_local=500, profits_distributed=3000, interest_rent_paid_outside=1500, taxes_local=800, taxes_outside=700),
        ]
        db.add_all(accs)

        # basket (living costs per FT worker for the month)
        basket = HouseholdCostBasket(period_start=pstart, period_end=pend, housing=1500, food=450, healthcare=350, childcare=0, transport=200, education=100, other=200)
        db.add(basket)

        # demography
        demo = Demography(period_start=pstart, period_end=pend, working_age_population=12000, participation_rate=0.65, standard_week_hours=40)
        db.add(demo)

        db.commit()
        return {"ok": True, "period_start": str(pstart), "period_end": str(pend)}
    finally:
        db.close()

# ------------------------
# Root
# ------------------------
@app.get("/")
def root():
    return {
        "name": "MarxTown MPI",
        "version": "0.1.0",
        "docs": "/docs",
        "hint": "POST /seed-demo then GET /metrics to try it."
    }
