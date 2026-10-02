# RetainAI Agent — Example Transcripts

Model: `us.amazon.nova-2-lite-v1:0`

## High risk, proactive outreach

**Customer message:** _(proactive outreach)_

**Tool calls:** get_customer_profile → score_churn_risk → recommend_offer

**Resolution:** `PROACTIVE_CHECK_IN` | **Offer:** `CONTRACT_UPGRADE` | **Escalate:** False

**Internal reasons:** tenure = 1, is_month_to_month = 1, InternetService = Fiber optic

> Thanks for being with us for 1 months. How can we help make your service better? If you’d like to switch to a 1-year contract, you could save $9.01 per month (about $108 over the year), bringing your monthly bill from $90.10 to $81.09.

## High risk, price complaint

**Customer message:** My bill keeps going up and the service hasn't changed. Why am I paying this much?

**Tool calls:** get_customer_profile → score_churn_risk → recommend_offer → search_policy

**Resolution:** `Our billing team will review this month's charges and reply within one business day. If there is an error, you will receive a credit of up to $25.` | **Offer:** `CONTRACT_UPGRADE` | **Escalate:** False

**Internal reasons:** tenure = 1, is_month_to_month = 1, InternetService = Fiber optic

> I see you’ve been with us just one month on a month-to-month plan, and your current bill is $72.10. We’ll have our billing team check this for you right away. If we find an error, you’ll get a credit of up to $25. To help keep costs steady, you might also consider switching to a 1-year contract, which would drop your monthly bill to $64.89—that’s a save of $7.21 each month.

## High risk, wants to close account

**Customer message:** Third outage this month. I want to close my account.

**Tool calls:** get_customer_profile → score_churn_risk → recommend_offer

**Resolution:** `A retention specialist will contact the customer within one business day.` | **Offer:** `CONTRACT_UPGRADE` | **Escalate:** True

**Internal reasons:** tenure = 1, is_month_to_month = 1, InternetService = Fiber optic

> I understand you're frustrated after three outages this month, and I see you've only been with us for one month on a month-to-month plan. A retention specialist will contact you within one business day. Switching to a 1-year contract would save you $8.46 per month, bringing your bill from $84.60 to $76.14, which you can consider once the outage issue is sorted.

## High risk, double charge

**Customer message:** You charged me twice this month and nobody has helped me.

**Tool calls:** get_customer_profile → score_churn_risk → recommend_offer → search_policy

**Resolution:** `Our billing team will review this month's charges and reply within one business day. If there is an error, you will receive a credit of up to $25.` | **Offer:** `CONTRACT_UPGRADE` | **Escalate:** True

**Internal reasons:** tenure = 1, is_month_to_month = 1, InternetService = Fiber optic

> I see you’ve been charged twice this month, and I’m sorry for the trouble. Our billing team will review your charges and get back to you within one business day—if there’s an error, you’ll get a credit of up to $25. To help keep your costs down once this is sorted, we can switch you to a 1-year contract with 10% off the monthly charge, saving you about $107 over the year.

## Medium risk, proactive

**Customer message:** _(proactive outreach)_

**Tool calls:** get_customer_profile → score_churn_risk → recommend_offer

**Resolution:** `PROACTIVE_CHECK_IN` | **Offer:** `CONTRACT_UPGRADE` | **Escalate:** False

**Internal reasons:** is_month_to_month = 1, InternetService = Fiber optic, PaymentMethod = Electronic check

> Thanks for 19 months with us. How can we make your Fiber service even better? If you’d like to lock in savings, switching to a year-long plan would cut your $96.80 monthly bill to $87.12 – that’s $9.68 off each month.

## Medium risk, simple question

**Customer message:** Where can I find my latest invoice?

**Tool calls:** get_customer_profile → score_churn_risk → search_policy → recommend_offer

**Resolution:** `Answer the question directly using the Customer FAQ from search_policy.` | **Offer:** `NO_OFFER` | **Escalate:** False

**Internal reasons:** is_month_to_month = 1, tenure = 2, PaymentMethod = Electronic check

> I see you’ve been with us for 2 months on a month-to-month plan. Your latest invoice can be found in your account portal under the ‘Billing’ section. If you need any help navigating there, just let me know!

## Low risk, proactive (should be NO_OFFER)

**Customer message:** _(proactive outreach)_

**Tool calls:** get_customer_profile → score_churn_risk → recommend_offer

**Resolution:** `Thank you for being with us for 7 months. How can we help make sure everything is working well with your $20.25 month-to-month service?` | **Offer:** `NO_OFFER` | **Escalate:** False

**Internal reasons:** is_month_to_month = 1, tenure = 7, PaperlessBilling = Yes

> Thanks for being with us for 7 months on your $20.25 month-to-month plan. How can we help make sure everything is working well for you?

## Low risk, simple question

**Customer message:** How do I update my payment method?

**Tool calls:** get_customer_profile → score_churn_risk → recommend_offer → search_policy

**Resolution:** `Answer the question directly using the Customer FAQ from search_policy.` | **Offer:** `NO_OFFER` | **Escalate:** False

**Internal reasons:** StreamingMovies = Yes, Contract = One year, OnlineSecurity = No

> I see you have been with us for 51 months and are on a one year plan. To update your payment method, please go to My Account, then Billing, and select Payment methods.

## Billing anomaly flagged

**Customer message:** My bill looks wrong this month.

**Tool calls:** get_customer_profile → score_churn_risk → recommend_offer → search_policy

**Resolution:** `Our billing team will review this month's charges and reply within one business day. If there is an error, the customer receives a credit of up to $25.` | **Offer:** `CONTRACT_UPGRADE` | **Escalate:** False

**Internal reasons:** is_month_to_month = 1, InternetService = Fiber optic, PaymentMethod = Electronic check

> I see your bill looks wrong this month, and you've been with us for 16 months. Our billing team will review this month's charges and reply within one business day—if there’s an error, you’ll get a credit of up to $25. To consider later, switching to a 1-year contract would save you $11.29 each month, bringing your bill from $112.95 to $101.66.

## Low risk, says they're leaving (override)

**Customer message:** I'm switching providers next month, please cancel my service.

**Tool calls:** get_customer_profile → score_churn_risk → recommend_offer

**Resolution:** `A retention specialist will contact the customer within one business day.` | **Offer:** `LOYALTY_DISCOUNT` | **Escalate:** True

**Internal reasons:** leaving, high risk by policy

> I see you’ve been with us for 20 months and are on a one year plan. A retention specialist will contact you within one business day. To keep your current setup, we can offer 10% off your $94.30 monthly charge for 6 months, saving you $57 in total—just let us know if you’d like to consider this while we work to resolve any concerns.

## Phone-only customer at risk

**Customer message:** _(proactive outreach)_

**Tool calls:** get_customer_profile → score_churn_risk → recommend_offer

**Resolution:** `Thank you for being with us for 1 months. How can we help make sure everything is working well for you?` | **Offer:** `CONTRACT_UPGRADE` | **Escalate:** False

**Internal reasons:** tenure = 1, is_month_to_month = 1, TotalCharges = 20.2

> Thanks for being with us for 1 months. If you’d like to lock in lower rates, we can switch you to a 1-year contract with 10% off the monthly charge. That would save you $2.02 each month (about $24 over the year), bringing your bill from $20.20 to $18.18.
