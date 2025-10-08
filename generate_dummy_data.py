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

 # Define possible values for categorical features (UPPERCASE matching production data)
 platforms = ['META', 'TIKTOK', 'INSTAGRAM']
 campaign_types = ['BAU', 'MM', 'FLOOD THE FEED']

 content_types = [
 'PAID - BRAND',
 'INFLUENCER - CFG - BOOSTED ONLY',
 'PAID - PARTNERSHIP',
 'OWNED - BOOSTED ONLY',
 'INFLUENCER - OGILVY - ORGANIC ONLY',
 'INFLUENCER - OGILVY - BOOSTED ONLY'
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
     if 'ORGANIC' in content_type:
         # Organic content has minimal spend
         total_spend = np.random.lognormal(5, 1) # Lower spend
     elif 'BOOSTED' in content_type or 'PAID' in content_type:
         # Boosted and Paid content have higher spend
         total_spend = np.random.lognormal(8, 1.5)
     else:
         total_spend = np.random.lognormal(7, 1)

     # Generate Impressions based on realistic patterns
     base_impressions = 50000 # Base impression count

     # Platform multipliers
     platform_multipliers = {
         'META': 2.5,
         'TIKTOK': 3.5,
         'INSTAGRAM': 2.0
     }

     # Campaign type effects
     campaign_multipliers = {'BAU': 1.0, 'MM': 1.2, 'FLOOD THE FEED': 1.8}

     # Content type effects
     content_multipliers = {
         'PAID - BRAND': 1.5,
         'INFLUENCER - CFG - BOOSTED ONLY': 2.0,
         'PAID - PARTNERSHIP': 1.4,
         'OWNED - BOOSTED ONLY': 1.2,
         'INFLUENCER - OGILVY - ORGANIC ONLY': 0.8,
         'INFLUENCER - OGILVY - BOOSTED ONLY': 1.6
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

     # Platform engagement multipliers (TIKTOK has highest engagement)
     platform_engagement_multipliers = {
         'TIKTOK': 1.5, # ~6% engagement rate
         'INSTAGRAM': 1.2, # ~4.8% engagement rate
         'META': 0.9 # ~3.6% engagement rate
     }

     # Content type engagement multipliers
     content_engagement_multipliers = {
         'INFLUENCER - CFG - BOOSTED ONLY': 1.6, # High engagement
         'INFLUENCER - OGILVY - ORGANIC ONLY': 1.4, # Good organic engagement
         'INFLUENCER - OGILVY - BOOSTED ONLY': 1.5,
         'OWNED - BOOSTED ONLY': 1.1,
         'PAID - PARTNERSHIP': 1.0,
         'PAID - BRAND': 0.8 # Lower engagement for paid ads
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
