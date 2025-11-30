# -----------------------------------------------------------
# Top 3 MATCHES TABLE — FIXED VERSION
# -----------------------------------------------------------
st.markdown("<div class='section-header'>Top 3 Matches</div>", unsafe_allow_html=True)

df = pd.DataFrame(results)

table_html = """
<style>
.match-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 10px;
    background: rgba(255, 255, 255, 0.80);
    backdrop-filter: blur(6px);
    border-radius: 12px;
    overflow: hidden;
    border: 1.5px solid #AFCBFF;
    box-shadow: 0 4px 14px rgba(0,0,0,0.05);
}
.match-table th {
    background-color: #167DFF;
    color: white;
    font-size: 18px;
    font-weight: 700;
    padding: 14px;
    text-align: center;
}
.match-table td {
    padding: 14px;
    text-align: center;
    font-size: 17px;
    color: #0E3A75;
    font-weight: 600;
}
</style>

<table class="match-table">
<tr>
    <th>Investor</th>
    <th>Match Score</th>
</tr>
"""

for _, row in df.iterrows():
    table_html += f"""
<tr>
    <td><strong>{row['investor']}</strong></td>
    <td><strong>{row['final']}</strong></td>
</tr>
"""

table_html += "</table>"

st.markdown(table_html, unsafe_allow_html=True)
