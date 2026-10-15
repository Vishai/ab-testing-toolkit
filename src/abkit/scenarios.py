"""Scenario registry.

Each scenario describes one A/B test: where its data lives, how its funnel is
ordered, what was (hypothetically) changed in variation B, and which metrics
play which role. The analysis code is scenario-agnostic and reads everything it
needs from here, so adding a new test means adding one entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    file: str
    funnel: tuple[str, ...]
    problem: str
    hypothesis: str
    change: str
    revenue_label: str | None = "Revenue"
    # Pre-treatment covariates: attributes fixed before a user sees A or B.
    # Only these are safe for regression adjustment and uplift modelling.
    covariates: tuple[str, ...] = ("Device", "UserType")
    # Page context recorded in the session. It can be shaped by the variation
    # itself (e.g. B moves people further down the funnel), so it is treated
    # as exploratory, never as an adjustment covariate.
    exploratory: tuple[str, ...] = ("PageViewed",)
    notes: list[str] = field(default_factory=list)

    @property
    def has_revenue(self) -> bool:
        return self.revenue_label is not None


SCENARIOS: dict[str, Scenario] = {
    "online_education": Scenario(
        key="online_education",
        title="Online Education - Course Enrolment",
        file="MIS5308_Scenario1_Online_Education_AB_Dataset.xlsx",
        funnel=("Course_Landing", "Course_Info", "Enrolment", "Registration"),
        problem=(
            "Low enrolment completion: weak click-through from course information "
            "to enrolment and high abandonment during registration."
        ),
        hypothesis=(
            "If we shorten registration to a single step and add a sticky 'Enrol now' "
            "CTA on course pages, then enrolment conversion will increase, because "
            "fewer steps and a persistent next action reduce friction at the two "
            "biggest drop-off points."
        ),
        change="Single-step registration + sticky enrolment CTA",
        revenue_label="Enrolment revenue",
    ),
    "travel_booking": Scenario(
        key="travel_booking",
        title="Travel Booking - Hotel Reservations",
        file="MIS5308_Scenario2_Travel_Booking_AB_Dataset.xlsx",
        funnel=("Search_Results", "Hotel_Details", "Booking", "Payment"),
        problem=(
            "Low booking completion: poor click-through from search results to "
            "booking pages and high abandonment in the booking/payment flow."
        ),
        hypothesis=(
            "If we show total price (incl. fees) on search results and collapse "
            "booking + payment into one page with express wallets, then booking "
            "conversion will increase, because price surprises and form length are "
            "the main causes of checkout abandonment."
        ),
        change="Transparent total pricing + one-page checkout with express pay",
        revenue_label="Booking revenue",
    ),
    "charity_donations": Scenario(
        key="charity_donations",
        title="Charity - Online Donations",
        file="MIS5308_Scenario3_Charity_Donations_AB_Dataset.xlsx",
        funnel=("Campaign", "Impact_Story", "Donation_Form", "Payment"),
        problem=(
            "Low donation completion: weak conversion from campaign pages and high "
            "abandonment on the donation form."
        ),
        hypothesis=(
            "If we add suggested donation amounts tied to concrete impact ('$50 = "
            "one week of meals') and trim the form to essentials, then donation "
            "completion will increase, because anchoring and tangible outcomes "
            "lower decision effort."
        ),
        change="Impact-anchored preset amounts + shorter donation form",
        revenue_label="Donation value",
    ),
    "free_trial": Scenario(
        key="free_trial",
        title="SaaS - Free-Trial Sign-up",
        file="MIS5308_Scenario4_Free_Trial_AB_Dataset.xlsx",
        funnel=("Product_Features", "Pricing", "Trial_Signup", "Account_Setup"),
        problem=(
            "Low free-trial sign-up rate: high exits from the pricing page and poor "
            "click-through from product features to sign-up."
        ),
        hypothesis=(
            "If we add a 'Start free - no card required' CTA on the pricing page and "
            "defer account setup until after first use, then trial sign-ups will "
            "increase, because perceived commitment is the main barrier at pricing."
        ),
        change="No-card trial CTA on pricing + deferred account setup",
        revenue_label=None,  # trial sign-ups carry no revenue in this dataset
    ),
}


def get(key: str) -> Scenario:
    try:
        return SCENARIOS[key]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Unknown scenario '{key}'. Options: {list(SCENARIOS)}") from exc
