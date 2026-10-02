# RetainAI Telco — Customer Retention Policy

This policy defines the only retention offers agents (human or AI) may make. Offers not listed here must never be promised.

## Offer catalog

The company has five offer codes. An agent may make at most one offer per customer contact.

- **ADDON_TRIAL**: three months free of one add-on service (Online Security, Online Backup, Device Protection, Tech Support, Streaming TV or Streaming Movies). The add-on is chosen by the recommendation model for that customer.
- **CONTRACT_UPGRADE**: move from month-to-month to a one-year contract with 10% off the monthly charge for the full year.
- **LOYALTY_DISCOUNT**: a percentage off the monthly charge for six months. The maximum percentage depends on revenue at risk and tenure (see Discount limits).
- **BILLING_REVIEW**: a free review of the customer's bill by the billing team, with a one-time credit of up to $25 if an error is confirmed.
- **NO_OFFER**: no incentive; a friendly check-in or a direct answer to the customer's question.

## Eligibility rules

ADDON_TRIAL requires an active internet service and that the customer does not already have the recommended add-on.
CONTRACT_UPGRADE is only available to customers currently on a month-to-month contract.
BILLING_REVIEW is only available when the billing anomaly detector has flagged the account.
Customers with low churn risk (probability below 0.30) receive NO_OFFER. Discounting customers who were not going to leave wastes margin.

## Discount limits

LOYALTY_DISCOUNT is capped by the customer's 12-month revenue at risk.
Up to 10% when revenue at risk is below $500.
Up to 15% when revenue at risk is $500 or more.
Up to 20% only when revenue at risk is $900 or more AND tenure is at least 12 months.
Discounts are never combined: a customer receives either CONTRACT_UPGRADE or LOYALTY_DISCOUNT, never both.

## Escalation to a human

Escalate to a human retention specialist when the customer explicitly asks to close their account or files a complaint AND their churn risk is high (probability 0.60 or above).
Also escalate any message that mentions legal action, regulators, or safety issues, regardless of risk.
When escalating, still draft a short, empathetic holding message that tells the customer a specialist will contact them within one business day.

## Communication guidelines

Messages to customers must be warm, specific to their situation, and under 120 words.
Never mention churn, risk scores, probabilities, models, segments, revenue at risk, or any internal data.
State offers exactly as defined in the catalog, including the exact discount percentage and duration.
Do not invent deadlines, prices, or extra perks.
Acknowledge the customer's problem before presenting any offer.
