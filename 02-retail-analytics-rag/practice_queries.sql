-- RETAIL ANALYTICS & POLICY ASSISTANT — SQL PRACTICE

-- 1. BASIC SELECT
SELECT *
FROM orders
LIMIT 10;

-- 2. SELECT + WHERE
-- TASK: Show orders from the North region with discount >= 10%.

-- 3. GROUP BY
-- TASK: Count total order lines by channel.

-- 4. AGGREGATES
-- TASK: Show region, total units sold, and average delivery days.

-- 5. GROUP BY + HAVING
-- TASK: Show customer segments with more than 100 order lines.

-- 6. ORDER BY
-- TASK: Find the 10 highest-value individual order lines.
-- order_value = quantity * unit_price * (1 - discount_pct)

-- 7. INNER JOIN
-- TASK: Join orders to products and show:
-- order_id, product_name, category, quantity, unit_price

-- 8. JOIN + GROUP BY
-- TASK: Calculate profit by category.
-- net_sales = quantity * unit_price * (1 - discount_pct)
-- cost      = quantity * unit_cost
-- profit    = net_sales - cost

-- 9. DATE THINKING
-- SQLite stores order_date as YYYY-MM-DD text.
-- substr(order_date, 1, 7) gives YYYY-MM.
-- TASK: Calculate monthly net sales.

-- 10. DATA QUALITY
-- TASK: Count NULL customer_segment and NULL rating values.
