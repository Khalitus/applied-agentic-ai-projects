-- Q1: escalation by issue type
SELECT
    issue_type,
    COUNT(*) AS ticket_count,
    SUM(escalated) AS escalated_count,
    ROUND(100.0 * SUM(escalated) / COUNT(*), 2) AS escalation_rate
FROM tickets
GROUP BY issue_type
ORDER BY escalation_rate DESC;


-- Q2: escalation by customer tier and product category
SELECT
    c.customer_tier,
    p.category,
    COUNT(*) AS ticket_count,
    ROUND(100.0 * SUM(t.escalated) / COUNT(*), 2) AS escalation_rate
FROM tickets AS t
LEFT JOIN customers AS c
    ON t.customer_id = c.customer_id
LEFT JOIN products AS p
    ON t.product_id = p.product_id
GROUP BY c.customer_tier, p.category
ORDER BY escalation_rate DESC;


-- Q3: rank issue types by escalation rate inside each region
WITH issue_stats AS (
    SELECT
        c.region,
        t.issue_type,
        COUNT(*) AS ticket_count,
        ROUND(100.0 * SUM(t.escalated) / COUNT(*), 2) AS escalation_rate
    FROM tickets AS t
    LEFT JOIN customers AS c
        ON t.customer_id = c.customer_id
    GROUP BY c.region, t.issue_type
)
SELECT
    region,
    issue_type,
    ticket_count,
    escalation_rate,
    RANK() OVER(
        PARTITION BY region
        ORDER BY escalation_rate DESC
    ) AS escalation_rank
FROM issue_stats
ORDER BY region, escalation_rank;


-- Q4: monthly ticket volume and escalation rate
SELECT
    STRFTIME('%Y-%m', created_at) AS month,
    COUNT(*) AS ticket_count,
    ROUND(100.0 * SUM(escalated) / COUNT(*), 2) AS escalation_rate
FROM tickets
GROUP BY month
ORDER BY month;


-- Q5: customers with at least 3 tickets and above-average escalation rate
WITH customer_stats AS (
    SELECT
        c.customer_id,
        c.customer_tier,
        COUNT(*) AS ticket_count,
        ROUND(100.0 * SUM(t.escalated) / COUNT(*), 2) AS escalation_rate
    FROM tickets AS t
    LEFT JOIN customers AS c
        ON t.customer_id = c.customer_id
    GROUP BY c.customer_id, c.customer_tier
)
SELECT
    customer_id,
    customer_tier,
    ticket_count,
    escalation_rate
FROM customer_stats
WHERE ticket_count >= 3
  AND escalation_rate > (
      SELECT 100.0 * AVG(escalated)
      FROM tickets
  )
ORDER BY escalation_rate DESC;