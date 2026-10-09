"""Admin analytics & result-analysis page (supports the report)."""

import pandas as pd
import plotly.express as px
import streamlit as st

from services import analytics_service


def render() -> None:
    """Result-analysis page: success rate, average score and report exports."""
    st.title("📈 Analytics & Result Analysis")

    analysis = analytics_service.result_analysis()

    row = st.columns(3)
    row[0].metric("Total queries", analysis["total_queries"])
    row[1].metric("Successful recommendations", analysis["successful_recommendations"])
    row[2].metric("Success rate", f"{analysis['success_rate']}%")

    row2 = st.columns(3)
    row2[0].metric("Avg recommendation score", analysis["average_recommendation_score"])
    row2[1].metric("Avg user rating", analysis["average_user_rating"])
    row2[2].metric("Recommendations made", analysis["total_recommendations_made"])

    st.divider()

    # Response time is recorded on every recommendation, so the report can
    # quote a real performance figure rather than an estimate.
    st.subheader("⏱️ Chatbot response time")
    timing = analytics_service.response_time_stats()
    if timing["measured_queries"]:
        speed = st.columns(4)
        speed[0].metric("Average", f"{timing['average_seconds']}s")
        speed[1].metric("Fastest", f"{timing['fastest_seconds']}s")
        speed[2].metric("Slowest", f"{timing['slowest_seconds']}s")
        speed[3].metric("Measured queries", timing["measured_queries"])
    else:
        st.caption(
            "No timings recorded yet - ask the chatbot for a recommendation "
            "and the figures will appear here."
        )

    st.divider()

    st.subheader("Recommendation success formula")
    st.latex(
        r"\text{Success Rate} = \frac{\text{successful recommendations}}"
        r"{\text{total queries}} \times 100"
    )

    st.subheader("Top recommended foods")
    top = analysis["top_recommended"]
    if top:
        df = pd.DataFrame(top)
        fig = px.bar(df, x="name", y="count", labels={"name": "Food", "count": "Recommendations"})
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption("No recommendations logged yet. Chat with the bot to generate data.")

    st.subheader("Highest rated foods")
    rated = analytics_service.highest_rated_foods(5)
    if rated:
        table = [
            {"Food": r["name"], "Avg rating": r["rating_avg"], "Ratings": r["rating_count"]}
            for r in rated
        ]
        st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)
    else:
        st.caption("No ratings yet.")

    st.divider()
    _export_section(analysis, rated)


def _export_section(analysis: dict, rated: list) -> None:
    """Download every table as CSV, ready to paste into the project report."""
    st.subheader("⬇️ Export for the report")
    st.caption("Each table downloads as a CSV you can drop straight into the report.")

    summary = pd.DataFrame(
        [
            {"Metric": "Total queries", "Value": analysis["total_queries"]},
            {
                "Metric": "Successful recommendations",
                "Value": analysis["successful_recommendations"],
            },
            {"Metric": "Success rate (%)", "Value": analysis["success_rate"]},
            {
                "Metric": "Average recommendation score",
                "Value": analysis["average_recommendation_score"],
            },
            {"Metric": "Average user rating", "Value": analysis["average_user_rating"]},
        ]
    )

    exports = [
        ("Result analysis summary", summary, "result_analysis"),
        ("Top recommended foods", pd.DataFrame(analysis["top_recommended"]), "top_recommended"),
        ("Highest rated foods", pd.DataFrame(rated), "highest_rated"),
        (
            "Food count by cuisine",
            pd.DataFrame(analytics_service.food_count_by_cuisine()),
            "food_by_cuisine",
        ),
        (
            "Ratings distribution",
            pd.DataFrame(analytics_service.ratings_distribution()),
            "ratings_distribution",
        ),
        (
            "User diet preferences",
            pd.DataFrame(analytics_service.user_diet_preferences()),
            "diet_preferences",
        ),
    ]

    columns = st.columns(3)
    for i, (label, frame, filename) in enumerate(exports):
        with columns[i % 3]:
            st.download_button(
                f"📄 {label}",
                data=frame.to_csv(index=False).encode("utf-8"),
                file_name=f"smartbite_{filename}.csv",
                mime="text/csv",
                width="stretch",
                disabled=frame.empty,
                key=f"export_{filename}",
            )
