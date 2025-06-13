import pandas as pd

tr = pd.read_csv("other_resources/calibration/raw_data/seattle-trips.csv", engine="pyarrow")
per = pd.read_csv("other_resources/calibration/raw_data/seattle-persons.csv", engine="pyarrow")

mapping = {
"Drive SOV": "SOV",
"Walk": "Non-Motorized",
"Drive HOV2": "HOV",
"Drive HOV3+": "HOV",
"Transit": "Transit",
"Bike": "Non-Motorized",
"Ride Hail": "Ride Hail",
"Other": "Other",
"Missing Response": "Other",
"School Bus": "Other",
"Micromobility": "Non-Motorized",
}

tr["simple-mode"] = tr["mode_class"].map(mapping)
tr["distance-bin"] = pd.cut(tr['distance_miles'],
       bins=[0, 0.5, 1.0, 2.0, 5.0, 10.0, 200.0],
       labels=["< 0.5 mi", "0.5 - 1 mi", "1 - 2 mi", "2 - 5 mi", "5 - 10 mi", "10+ mi"],)

tr_sub = tr.loc[tr['travel_dow'].isin(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])]

trip_counts = tr_sub.groupby(["distance-bin","simple-mode"])["trip_weight"].agg(sum).unstack() / per.loc[per.person_id.isin(tr_sub.person_id.unique()),"person_weight"].sum()
trip_counts.to_csv("other_resources/calibration/calibration-data/trip_counts_by_mode_and_distance_per_capita.csv")