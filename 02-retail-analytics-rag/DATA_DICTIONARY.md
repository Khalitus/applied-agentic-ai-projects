# Data Dictionary

## retail_orders.csv

| Column | Meaning |
|---|---|
| order_id | Unique synthetic order-line ID |
| order_date | YYYY-MM-DD |
| region | South, North, Central, West |
| city | Synthetic city |
| channel | Online, Retail Store, Corporate |
| customer_segment | Consumer, SME, Enterprise, or intentional NULL |
| product_id | Foreign key to products.csv |
| quantity | Units |
| unit_price | Price per unit before discount |
| discount_pct | 0.00 / 0.05 / 0.10 / 0.15 |
| delivery_days | Delivery duration |
| rating | 1–5 or intentional NULL |
| returned | Yes / No |
| payment_method | Payment type |

## products.csv

| Column | Meaning |
|---|---|
| product_id | Product key |
| product_name | Fictional product |
| category | Product category |
| list_price | Reference price |
| unit_cost | Synthetic cost |
| supplier | Fictional supplier |
| warranty_months | Warranty |

## Derived measures

```text
gross_sales = quantity * unit_price
net_sales   = quantity * unit_price * (1 - discount_pct)
cost        = quantity * unit_cost
profit      = net_sales - cost
```
