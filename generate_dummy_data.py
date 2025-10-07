import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

def generate_dummy_campaign_data(n_samples=1000):
 """
 Generate dummy social media campaign data for testing the ML pipeline.

 UPDATED SCHEMA:
 - Target variables: Impressions AND Engagement (multi-output)
 - Platforms: Meta, TikTok, Instagram (3 platforms only)
 - Features: campaign_type, Platform, content_type, total_spend (Quarter REMOVED)
 - Engagement = aggregate of (Likes + Shares + Comments + Saves)
 - No Content_Theme, Content_Details, OPE, Influencer_Name, Spend_Production, Spend_Media
 """

 # Define possible values for categorical features
 platforms = ['Meta', 'TikTok', 'Instagram']
 campaign_types = ['Bau', 'Mm', 'Flood The Feed']

 content_types = [
 'Paid - Brand',
 'Influencer - Cfg - Boosted Only',
 'Paid - Partnership',
 'Owned - Boosted Only',
 'Influencer - Ogilvy - Organic Only',
 'Influencer - Ogilvy - Boosted Only'
 ]

 # Generate the dataset
 data = []

 for i in range(n_samples):
 # Basic campaign features
 platform = random.choice(platforms)
 campaign_type = random.choice(campaign_types)
 content_type = random.choice(content_types)

 # Total spend variable
 # Different content types have different spend patterns
 if 'Organic' in content_type:
 # Organic content has minimal spend
 total_spend = np.random.lognormal(5, 1) # Lower spend
 elif 'Boosted' in content_type or 'Paid' in content_type:
 # Boosted and Paid content have higher spend
 total_spend = np.random.lognormal(8, 1.5)
 else:
 total_spend = np.random.lognormal(7, 1)

 # Generate Impressions based on realistic patterns
 base_impressions = 50000 # Base impression count

 # Platform multipliers
 platform_multipliers = {
 'Meta': 2.5,
 'TikTok': 3.5,
 'Instagram': 2.0
 }

 # Campaign type effects
 campaign_multipliers = {'Bau': 1.0, 'Mm': 1.2, 'Flood The Feed': 1.8}

 # Content type effects
 content_multipliers = {
 'Paid - Brand': 1.5,
 'Influencer - Cfg - Boosted Only': 2.0,
 'Paid - Partnership': 1.4,
 'Owned - Boosted Only': 1.2,
 'Influencer - Ogilvy - Organic Only': 0.8,
 'Influencer - Ogilvy - Boosted Only': 1.6
 }

 # Calculate impressions with some randomness
 impressions_multiplier = (
 platform_multipliers.get(platform, 1.0) *
 campaign_multipliers.get(campaign_type, 1.0) *
 content_multipliers.get(content_type, 1.0) *
 (1 + np.log(total_spend + 1) / 12) # Spend effect with diminishing returns
 )

 impressions = int(base_impressions * impressions_multiplier * np.random.lognormal(0, 0.6))
 impressions = max(1000, impressions) # Minimum 1000 impressions

 # Generate Engagement based on Impressions
 # Engagement = aggregate of (Likes + Shares + Comments + Saves)
 # Industry standard: 2-8% engagement rate, varies by platform and content

 base_engagement_rate = 0.04 # 4% base rate

 # Platform engagement multipliers (TikTok has highest engagement)
 platform_engagement_multipliers = {
 'TikTok': 1.5, # ~6% engagement rate
 'Instagram': 1.2, # ~4.8% engagement rate
 'Meta': 0.9 # ~3.6% engagement rate
 }

 # Content type engagement multipliers
 content_engagement_multipliers = {
 'Influencer - Cfg - Boosted Only': 1.6, # High engagement
 'Influencer - Ogilvy - Organic Only': 1.4, # Good organic engagement
 'Influencer - Ogilvy - Boosted Only': 1.5,
 'Owned - Boosted Only': 1.1,
 'Paid - Partnership': 1.0,
 'Paid - Brand': 0.8 # Lower engagement for paid ads
 }

 # Calculate engagement rate with variability
 engagement_rate = (
 base_engagement_rate *
 platform_engagement_multipliers.get(platform, 1.0) *
 content_engagement_multipliers.get(content_type, 1.0) *
 np.random.lognormal(0, 0.3) # Add realistic noise
 )

 # Clip engagement rate to realistic bounds (0.5% - 15%)
 engagement_rate = np.clip(engagement_rate, 0.005, 0.15)

 # Calculate aggregate engagement (Likes + Shares + Comments + Saves)
 engagement = int(impressions * engagement_rate)
 engagement = max(50, engagement) # Minimum 50 engagements

 data.append({
 'campaign_type': campaign_type,
 'Platform': platform,
 'content_type': content_type,
 'total_spend': round(total_spend, 2),
 'Impressions': impressions,
 'Engagement': engagement
 })

 return pd.DataFrame(data)

if __name__ == "__main__":
 # Generate dummy data
 print("Generating dummy social media campaign data...")
 df = generate_dummy_campaign_data(1000)

 # Save to CSV
 output_path = "data/campaign_data.csv"
 df.to_csv(output_path, index=False)
 print(f"Dummy data saved to {output_path}")

 # Display basic statistics
 print(f"\nDataset shape: {df.shape}")
 print(f"Missing values per column:")
 print(df.isnull().sum())
 print(f"\nFirst few rows:")
 print(df.head())

 print(f"\nImpressions statistics:")
 print(df['Impressions'].describe())

 print(f"\nEngagement statistics:")
 print(df['Engagement'].describe())

 print(f"\nEngagement Rate statistics:")
 df['Engagement_Rate'] = df['Engagement'] / df['Impressions']
 print(df['Engagement_Rate'].describe())
 print(f"Mean Engagement Rate: {df['Engagement_Rate'].mean():.2%}")

 print(f"\nPlatform distribution:")
 print(df['Platform'].value_counts())

 print(f"\nContent type distribution:")
 print(df['content_type'].value_counts())

 print(f"\nEngagement Rate by Platform:")
 print(df.groupby('Platform')['Engagement_Rate'].mean().sort_values(ascending=False))

 print(f"\nEngagement Rate by Content Type:")
 print(df.groupby('content_type')['Engagement_Rate'].mean().sort_values(ascending=False))
