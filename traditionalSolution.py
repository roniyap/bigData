import pandas as pd
from timeit import default_timer as timer
import tracemalloc
import psutil

def reduce_memory(df):
    #Reduce memory usage of DataFrame by downcasting numeric columns
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df

df_games = reduce_memory(pd.read_csv("../PythonProject/games.csv"))
df_games["date_release"] = df_games["date_release"].astype(str).str[:4].astype(int)
df_games.rename(columns={"date_release": "year_release"}, inplace=True)
df_games.drop(["win", "mac", "linux", "steam_deck"], axis=1, inplace=True)

file_path = "../PythonProject/recommendations.csv"
chunksize = 10000
chunks = []

for chunk in pd.read_csv(file_path, chunksize=chunksize):
    chunk = reduce_memory(chunk)
    chunks.append(chunk)

df_recs = pd.concat(chunks, ignore_index=False)
df_users = reduce_memory(pd.read_csv("../PythonProject/users.csv"))
meta_data = reduce_memory(pd.read_json("../PythonProject/games_metadata.json"))
meta_data["tags"] = meta_data["tags"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)


#Task 1: For analysing pricing trends we can find the average price of games by year of release:
tracemalloc.start()
start = timer()


avg_price = df_games.groupby("year_release")["price_final"].agg("mean")
print(avg_price.head(10))

end = timer()
current, peak = tracemalloc.get_traced_memory()
print(f"Task 1 - Memory: Current = {current / 1024:.2f} KB, Peak = {peak / 1024:.2f} KB")
print(f"Task 1 - Time: {end - start:.2f} sec")
tracemalloc.stop()


#Task 2: Find the most common genres in the dataset to see which genres are trending:
tracemalloc.start()
start = timer()

genres = []
genre_count = {}

for g in meta_data["tags"]:
    if pd.isna(g) or g.strip() == "":
        continue
    for genre in g.split(","):
        genre = genre.strip()
        if genre not in genres:
            genres.append(genre)
            genre_count[genre] = 0
        genre_count[genre] += 1

sorted_genre_count = dict(sorted(genre_count.items(), key=lambda item: item[1], reverse=True))
for genre, count in list(sorted_genre_count.items())[:10]:
    print(f"{genre}: {count}")

end = timer()
current, peak = tracemalloc.get_traced_memory()
print(f"Task 2 - Memory: Current = {current / 1024:.2f} KB, Peak = {peak / 1024:.2f} KB")
print(f"Task 2 - Time: {end - start:.2f} sec")
tracemalloc.stop()


#Task 3: We could identify which price points generate the most recommendations:
tracemalloc.start()
start = timer()

df_merged = df_games.merge(df_recs, on="app_id", how="inner")
bins = [0, 10, 20, 30, 40, 50, 60, 61]
labels = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60+"]

df_merged["Price"] = pd.cut(df_merged["price_final"], bins=bins, labels=labels)
df_price_recommendations = df_merged[df_merged["is_recommended"] == True] \
    .groupby("Price")["app_id"].count().reset_index().rename(columns={"app_id": "count"})
df_price_recommendations = df_price_recommendations.sort_values(by="count", ascending=False)

print(df_price_recommendations.head(10))

end = timer()
current, peak = tracemalloc.get_traced_memory()

print(f"Task 3 - Memory: Current = {current / 1024**2:.2f} MB, Peak = {peak / 1024**2:.2f} MB")
print(f"Task 3 - Time: {end - start:.2f} sec")
tracemalloc.stop()


#Task 4:We could find the most recommended genres this would be computationally heavy since it joins millions of records
start = timer()
mem_before = psutil.virtual_memory().used / 1024**2

genres = []
genre_count = {}
df_merged2 = meta_data.merge(df_recs, on="app_id", how="inner")

for row in df_merged2.itertuples():
    if row.is_recommended:
        if pd.isna(row.tags) or row.tags.strip() == "":
            continue
        for genre in row.tags.split(","):
            genre = genre.strip()
            if genre not in genres:
                genres.append(genre)
                genre_count[genre] = 0
            genre_count[genre] += 1

sorted_genre_count = dict(sorted(genre_count.items(), key=lambda item: item[1], reverse=True))

for genre, count in list(sorted_genre_count.items())[:10]:
    print(f"{genre}: {count}")

end = timer()
mem_after = psutil.virtual_memory().used / 1024**2
print(f"Task 4 - Memory: Used = {mem_after - mem_before:.2f} MB")
print(f"Task 4 - Time: {end - start:.2f} sec")


#Task 5: For analysing user reviews we can calculate the percentage of positive reviews per game
tracemalloc.start()
start = timer()

df_pos_reviews = (df_merged.groupby("title")["is_recommended"].sum()
                  .reset_index().rename(columns={"is_recommended": "positive_review_count"}))

total_reviews_per_game = df_merged.groupby("title")["is_recommended"].count()
df_pos_reviews = df_pos_reviews[df_pos_reviews["title"].map(total_reviews_per_game) > 500]
df_pos_reviews["positive_percentage"] = (df_pos_reviews["positive_review_count"] /
                                         df_pos_reviews["title"].map(total_reviews_per_game) * 100)
df_pos_reviews = df_pos_reviews.sort_values(by="positive_percentage", ascending=False)

print(df_pos_reviews.head(10))

end = timer()
current, peak = tracemalloc.get_traced_memory()
print(f"Task 5 - Memory: Current = {current / 1024:.2f} KB, Peak = {peak / 1024:.2f} KB")
print(f"Task 5 - Time: {end - start:.2f} sec")
tracemalloc.stop()
