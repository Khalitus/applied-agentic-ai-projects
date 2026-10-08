from src.data_loader import load_claims


def main():
    claims = load_claims()

    print("\nDataset overview")
    print(f"Rows: {claims.shape[0]}")
    print(f"Columns: {claims.shape[1]}")

    print("\nColumn types")
    print(claims.dtypes)

    print("\nMissing values")
    print(claims.isna().sum())

    print("\nDuplicate claim IDs")
    print(claims["claim_id"].duplicated().sum())

    print("\nNumber of each Incident Type")
    print(claims["incident_type"].value_counts())

    print("\nAverage Claim by Severity")
    print(claims["claim_amount"].groupby(claims["damage_severity"]).mean())

    print("\nData Integrity")
    print((claims['policy_start_date'] > claims['incident_date']).sum())

    print("\nTarget summary")
    print(claims["claim_amount"].describe())


if __name__ == "__main__":
    main()