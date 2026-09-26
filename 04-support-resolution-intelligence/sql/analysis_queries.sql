-- Support Resolution Intelligence
-- Write these queries yourself. Keep them readable and validate the result shape.

-- Q1: Escalation rate by issue type.
-- Expected columns: issue_type, ticket_count, escalated_count, escalation_rate


-- Q2: Escalation rate by customer tier and product category.
-- Requires tickets -> customers -> products joins.
-- Expected columns: customer_tier, category, ticket_count, escalation_rate


-- Q3: Rank issue types by escalation rate inside each region.
-- Use a CTE plus a window function.
-- Expected columns: region, issue_type, ticket_count, escalation_rate, escalation_rank


-- Q4: Monthly ticket volume and escalation rate.
-- Expected columns: month, ticket_count, escalation_rate


-- Q5: Find customers with >= 3 tickets and above-average escalation rate.
-- Expected columns: customer_id, customer_tier, ticket_count, escalation_rate
