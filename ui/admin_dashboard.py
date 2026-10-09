"""Admin dashboard with metric cards and charts."""

import pandas as pd
import plotly.express as px
import streamlit as st

from services import analytics_service, user_service


def render() -> None:
    """Admin landing page: headline metrics and the four dashboard charts."""
    st.title("📊 Admin Dashboard")

    metrics = analytics_service.overview_metrics()
    row1 = st.columns(3)
    row1[0].metric("Total users", metrics["total_users"])
    row1[1].metric("Total foods", metrics["total_foods"])
    row1[2].metric("Total recommendations", metrics["total_recommendations"])

    row2 = st.columns(3)
    row2[0].metric("Average rating", metrics["average_rating"])
    row2[1].metric("Most recommended", metrics["most_recommended_food"])
    row2[2].metric("Popular cuisine", metrics["popular_cuisine"])

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 5 recommended foods")
        top = analytics_service.top_recommended_foods(5)
        if top:
            df = pd.DataFrame(top)
            fig = px.bar(df, x="name", y="count", labels={"name": "Food", "count": "Times recommended"})
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption("No recommendations logged yet.")

    with col2:
        st.subheader("Food count by cuisine")
        cuisines = analytics_service.food_count_by_cuisine()
        if cuisines:
            df = pd.DataFrame(cuisines)
            fig = px.pie(df, names="cuisine", values="count")
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption("No foods yet.")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Ratings distribution")
        dist = analytics_service.ratings_distribution()
        df = pd.DataFrame(dist)
        fig = px.bar(df, x="rating", y="count", labels={"rating": "Stars", "count": "Count"})
        st.plotly_chart(fig, width="stretch")

    with col4:
        st.subheader("User diet preferences")
        diets = analytics_service.user_diet_preferences()
        if diets:
            df = pd.DataFrame(diets)
            df["diet_type"] = df["diet_type"].str.replace("_", " ").str.title()
            fig = px.bar(df, x="diet_type", y="count", labels={"diet_type": "Diet", "count": "Users"})
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption("No diet preferences recorded yet.")

    st.divider()
    st.subheader("👥 All users")
    users = user_service.list_users()
    if users:
        table = [
            {
                "Name": u.get("name"),
                "Email": u.get("email"),
                "Role": u.get("role"),
                "Active": u.get("is_active", True),
            }
            for u in users
        ]
        st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)
