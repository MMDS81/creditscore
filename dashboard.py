# ceci est exemple échantillon de 200 individus

import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
import shap
from fastapi import FastAPI
# importer les données
def main():
    @st.cache
    def chargement_data():
        data = pd.read_csv('data.csv')
        target = pd.read_csv('target.csv')
        data.drop('Unnamed: 0', axis=1, inplace=True)
        target.drop('Unnamed: 0', axis=1, inplace=True)

        # echantillonnage
        data = data.head(200)
        target = target.head(200)

    return data, target

    def charger_model():
        pickle_in = open('mypicklefile', 'rb')
        clf = pickle.load(pickle_in)
        return clf

    def identite_client(data, id):
        data_client = data[data['SK_ID_CURR'] == int(id)]
        return data_client

    def load_age_population(data):
        data_age = round((data["DAYS_BIRTH"] / -365), 2)
        return data_age

    def load_income_population(data):
        data_revenu = data[data['AMT_INCOME_TOTAL'] < 350000]
        return data_revenu

    def load_children_population(data):
        data_children = data[data['CNT_CHILDREN'] < 5]
        return data_children

    # récupérer la liste des identifiants des clients
    list_ids = data['SK_ID_CURR'].tolist()

    # rajouter la liste des clients à une liste déroulante
    st.sidebar.header("L'identifiant du client :")
    id = st.sidebar.selectbox("Veuillez choisir votre numéro d'identifiiant", list_ids)
    st.sidebar.write('Votre id_client est:', id)

    def fetch(session, url):
        try:
            result = session.get(url)
            return result.json()
        except Exception:
            return {}

    with st.sidebar.form(" "):
        st.sidebar.columns(2)
        submitted = st.form_submit_button("prediction")

    # appel à l'api de prediction
    session = requests.Session()
    if submitted:
        probabilite = fetch(session, f"https://creditscoring.herokuapp.com/predict/{id}")
        st.sidebar.write("**Probabilité :**", round(float(probabilite) * 100, 2), "**%**")
        if (float(probabilite) < 0.5):
            decision = "<font color='green'>**Prêt accordé!**</font>"
        else:
            decision = "<font color='red'>**Prêt rejeté!**</font>"
            st.sidebar.write("**Decision :**", decision, unsafe_allow_html=True)

    # mise en forme du titre
    html_temp = """
    <div style="background-color: tomato; padding:10px; border-radius:10px">
    <h1 style="color: white; text-align:center">Dashboard Scoring Credit</h1>
    </div>
    <p style="font-size: 20px; font-weight: bold; text-align:center">Credit decision support…</p>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.markdown('##')
    st.markdown('##')

    # afficher les données de l'utilisateur
    st.write(' **Veuillez consulter vos données :**')
    infos_client = identite_client(data, id)
    st.write(identite_client(data, id))

    st.markdown('##')
    st.markdown('##')

    # afficher les distributions des principaux features
    st.write(' **Veuillez cocher les données dont vous voulez voir la distribution :**')
    # distribution d'age

    if st.checkbox("Age"):
        data_age = load_age_population(data)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_age, edgecolor='k', color="goldenrod", bins=20)
        ax.axvline(int(infos_client["DAYS_BIRTH"].values / -365), color="green", linestyle='--')
        ax.set(title='Age du client', xlabel='Age(année)', ylabel='')
        st.pyplot(fig)

    # distribution des revenus
    if st.checkbox("Revenues"):
        data_revenu = load_income_population(data)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_revenu["AMT_INCOME_TOTAL"], edgecolor='k', color="goldenrod", bins=10)
        ax.axvline(int(infos_client["AMT_INCOME_TOTAL"].values[0]), color="green", linestyle='--')
        ax.set(title='Revenues des clients', xlabel='Revenues (USD)', ylabel='')
        st.pyplot(fig)

    # distribution du nombre d'enfants
    if st.checkbox("Nombre d'enfants"):
        data_children = load_children_population(data)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_children["CNT_CHILDREN"], edgecolor='k', color="goldenrod", bins=20)
        ax.axvline(int(data_children["CNT_CHILDREN"].values[0]), color="green", linestyle='--')
        ax.set(title="Nombre d'enfants des clients", xlabel="Nombre d'enfants", ylabel='')
        st.pyplot(fig)

    # afficher les relations entre variables
    if st.checkbox("consulter les relations entre variables"):
        data_sk = data.reset_index(drop=False)
        data_sk.DAYS_BIRTH = (data_sk['DAYS_BIRTH'] / -365).round(1)
        fig, ax = plt.subplots(figsize=(10, 10))
        fig = px.scatter(data_sk, x='DAYS_BIRTH', y="AMT_INCOME_TOTAL",
                         size="AMT_INCOME_TOTAL", color='CODE_GENDER',
                         hover_data=['NAME_FAMILY_STATUS', 'CNT_CHILDREN', 'NAME_CONTRACT_TYPE', 'SK_ID_CURR'])

        fig.update_layout({'plot_bgcolor': '#f0f0f0'},
                          title={'text': "Relation Age / Income Total", 'x': 0.5, 'xanchor': 'center'},
                          title_font=dict(size=20, family='Verdana'), legend=dict(y=1.1, orientation='h'))

        fig.update_traces(marker=dict(line=dict(width=0.5, color='#3a352a')), selector=dict(mode='markers'))
        fig.update_xaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                         title="Age", title_font=dict(size=18, family='Verdana'))
        fig.update_yaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                         title="Income Total", title_font=dict(size=18, family='Verdana'))
        st.plotly_chart(fig)

        data_children = load_children_population(data)
        fig2, ax = plt.subplots(figsize=(10, 10))
        fig2 = px.scatter(data_children, x='CNT_CHILDREN', y="AMT_INCOME_TOTAL",
                          size="AMT_INCOME_TOTAL", color='CODE_GENDER',
                          hover_data=['NAME_FAMILY_STATUS', 'CNT_CHILDREN', 'NAME_CONTRACT_TYPE', 'SK_ID_CURR'])

        fig2.update_layout({'plot_bgcolor': '#f0f0f0'},
                           title={'text': "Relation nombre des enfants / Income Total", 'x': 0.5, 'xanchor': 'center'},
                           title_font=dict(size=20, family='Verdana'), legend=dict(y=1.1, orientation='h'))

        fig2.update_traces(marker=dict(line=dict(width=0.5, color='#3a352a')), selector=dict(mode='markers'))
        fig2.update_xaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                          title="nbre enfants", title_font=dict(size=18, family='Verdana'))
        fig.update_yaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                         title="Income Total", title_font=dict(size=18, family='Verdana'))
        st.plotly_chart(fig2)

        # feature importance locale
        st.markdown('##')
        st.markdown('##')
        st.write(' **Les données qui ont influencé la décision prise :**')
    # compute shap values

    if st.checkbox("Consulter "):
        st.write(
            "**Description :** vous êtes en risque de refus à cause des données marquées en rouge. Ceux marquées en bleu favorisent l'acceptation de votre demande.")
        shap.initjs()
        X = data
        X = X[X.index == id]
        number = st.slider("Veuillez sélectionner le nombre de features …", 0, 20, 5)
    # afficher le graphe de feature importance local
    model = load_model()
    explainer = shap.Explainer(model, data)
    shap_values = explainer(data)
    fig, ax = plt.subplots(figsize=(10, 10))
    shap.plots.bar(shap_values[0], max_display=number)
    st.pyplot(fig)


if __name__ == '__main__':
    main()
