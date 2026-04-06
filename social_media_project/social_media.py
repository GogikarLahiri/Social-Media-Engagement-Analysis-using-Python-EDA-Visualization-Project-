import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("social_media_engagement1.csv")

print(data.head())
print("\nDataset Information:")
print(data.info())

print("\nDataset Shape:")
print(data.shape)
print("\nMissing Values:")
print(data.isnull().sum())
data = data.drop_duplicates()

print("\nShape After Removing Duplicates:")
print(data.shape)
data["post_time"] = pd.to_datetime(data["post_time"])
data["hour"] = data["post_time"].dt.hour
data["month"] = data["post_time"].dt.month
data["total_engagement"] = data["likes"] + data["comments"] + data["shares"]

print("\nDataset With Total Engagement:")
print(data.head())
top_posts = data.sort_values(by="total_engagement", ascending=False).head(10)

print("\nTop 10 Posts:")
print(top_posts[["post_id","platform","post_type","total_engagement"]])
platform_engagement = data.groupby("platform")["total_engagement"].sum()

print("\nEngagement by Platform:")
print(platform_engagement)
post_type_engagement = data.groupby("post_type")["total_engagement"].sum()

print("\nEngagement by Post Type:")
print(post_type_engagement)
day_engagement = data.groupby("post_day")["total_engagement"].sum()

print("\nEngagement by Day:")
print(day_engagement)
hour_engagement = data.groupby("hour")["total_engagement"].mean()

print("\nAverage Engagement by Hour:")
print(hour_engagement)
sentiment_analysis = data.groupby("sentiment_score")["total_engagement"].mean()

print("\nSentiment vs Engagement:")
print(sentiment_analysis)
data["engagement_rate"] = data["total_engagement"] / data["likes"]

print("\nEngagement Rate Example:")
print(data[["post_id","engagement_rate"]].head())
platform_engagement.plot(kind="bar")

plt.title("Total Engagement by Platform")
plt.xlabel("Platform")
plt.ylabel("Total Engagement")

plt.show()
post_type_engagement.plot(kind="bar")

plt.title("Engagement by Post Type")
plt.xlabel("Post Type")
plt.ylabel("Total Engagement")

plt.show()
day_engagement.plot(kind="bar")

plt.title("Best Posting Day")
plt.xlabel("Day")
plt.ylabel("Total Engagement")

plt.show()
data.to_csv("clean_social_media_data.csv", index=False)
