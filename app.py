from plotly.subplots import make_subplots

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

data_filename = 'data/results.parquet'
run_type = 'Final'
n_rows = 20

def load_data(data_filename):
    df = pd.read_parquet(data_filename)
    df['best_split_rank'] = df[['split_1_rank','split_2_rank', 'split_3_rank', 'split_4_rank', 'finish_rank']].min(axis=1)

    return df

def build_dot_chart(df, category, timing_column):
    
    chart_data = df.sort_values(by=f'{timing_column}_gap',ascending=False)

    dot_chart = go.Scatter(x=chart_data[f'{timing_column}_gap'], 
                            y=chart_data['name'].str.title(),
                            mode="markers+text",
                            text=chart_data[f'{timing_column}_gap']) 

    if category == 'Elite Female':
        height = 400
    else:
        height = 700
    width = 300

    fig = go.Figure(dot_chart)
    fig.update_layout(width=width, height=height, showlegend=False, title=timing_column.replace('_', ' ').capitalize(), margin=dict(r=20))
    fig.update_xaxes(autorange="reversed", automargin=True)
    fig.update_traces(textposition='middle right', cliponaxis=False)

    return fig


def build_rank_bump_chart(df, category):
    df = df[['name', 'split_1_rank','split_2_rank', 'split_3_rank', 'split_4_rank', 'finish_rank']]
    field_size = len(df)
    df = pd.melt(df, id_vars=['name'], var_name='split', value_name='value').rename(columns={'value':'position'})
    df['split'] = df['split'].str.replace('_', ' ').str.replace(' rank', '')

    if category == 'Elite Female':
        height = 400
    else:
        height = 900
    width = 400

    fig = px.line(df, x='split', y='position', color='name', width=width, height=height, markers=True, title='Position by split')   
    fig.update_layout(margin=dict(r=100), showlegend=False)
    fig.update_yaxes(range=[field_size, 1])
    fig.update_traces(textposition='middle right', cliponaxis=False)

    return fig


def build_sector_heatmap(df, category):

    df = df[['name', 'sector_1_gap', 'sector_2_gap', 'sector_3_gap', 'sector_4_gap', 'sector_5_gap','finish_gap']].sort_values(by='finish_gap', ascending=False, na_position='first')
    # print(df)
    
    columns = ['sector_1_gap', 'sector_2_gap', 'sector_3_gap', 'sector_4_gap', 'sector_5_gap']
    label_data = df[columns]
    colour_data = df[columns].clip(upper=5.0)
    names = df['name']

    if category == 'Elite Female':
        height = 400
    else:
        height = 900

    fig = go.Figure(go.Heatmap(z=colour_data.values,
                    x=['Sector 1', 'Sector 2', 'Sector 3', 'Sector 4', 'Sector 5'],
                    y=names.values,
                    text=label_data.values,
                    texttemplate="+%{text}",
                    colorscale='Reds_r'))
   
    fig.update_layout(title='Sector times heatmap',
                    xaxis=dict(side='top'),
                    width=400,
                    height=height)
    
    fig.update_traces(showscale=False)
    
    return fig

# def make_season_points_plot(df):
#     runs = df[['round_number', 'run_type']].drop_duplicates()
#     runs['run_order'] = runs['run_type'].map({'Qualifying':'1', 'Semi-final':'2', 'Final':'3'})
#     runs['run_type_short'] = runs['run_type'].map({'Qualifying':'Q', 'Semi-final':'SF', 'Final':'F'})
#     runs['run_type_concat'] = runs['round_number'].astype('str') + '_' + runs['run_order'].astype('str') + '_' + runs['run_type_short']
#     racers = df[df['category'] == 'Elite Male'][['name', 'category']].drop_duplicates()


#     points = racers.merge(runs, how='cross')
#     points = points.merge(df[['name', 'round_number', 'run_type', 'finish_rank', 'points']], left_on=['name', 'round_number', 'run_type'], right_on=['name', 'round_number', 'run_type'], how='left')
#     points = points.fillna(0)

#     points['points_rolling'] = points.sort_values(by=['name', 'round_number', 'run_order']).groupby('name')['points'].cumsum()

#     top_20 = points.groupby('name')['points'].sum().sort_values(ascending=False)[:20].index
#     chart_data = points[points['name'].isin(top_20)].sort_values(by=['run_type_concat'])

#     fig = px.line(chart_data, x=['run_type_concat'], y='points_rolling', color='name')

#     return fig

### APP DEFINITION ###

# Title of the app
st.title("LOOK AT THE STATS!!!")

#Load dataset and apply filters
df = load_data(data_filename)

st.write('Select a race and a category to see how the race went down.')

selected_race = st.selectbox('Choose a race:', df['round_number_venue'].drop_duplicates().sort_values())
selected_category = st.selectbox('Choose a category:', df['category'].drop_duplicates().unique())

race_df = df[(df['round_number_venue'] == selected_race) & (df['category'] == selected_category) & (df['run_type'] == run_type) & (pd.notnull(df['split_1_rank']))]

# Some code to build the heatmap/bumpchart plot. Needs to get tidied up into a function
bump_chart = go.Figure(build_rank_bump_chart(race_df, selected_category))
heatmap = go.Figure(build_sector_heatmap(race_df, selected_category))

bump_traces = []
heatmap_traces = []
for trace in range(len(bump_chart["data"])):
    bump_traces.append(bump_chart["data"][trace])
for trace in range(len(heatmap["data"])):
    heatmap_traces.append(heatmap["data"][trace])

#Create a 1x2 subplot
bump_heatmap = make_subplots(rows=1, cols=2, horizontal_spacing=0.2,  subplot_titles=("Positions by split", "Sector time gaps")) 

# Get the Express fig broken down as traces and add the traces to the proper plot within in the subplot
for traces in bump_traces:
    bump_heatmap.append_trace(traces, row=1, col=1)
for traces in heatmap_traces:
    bump_heatmap.append_trace(traces, row=1, col=2)

bump_heatmap['layout']['yaxis']['range'] = [len(race_df), 0.5]
bump_heatmap.update_layout(showlegend=False) 


if selected_category == 'Elite Male':
    bump_heatmap.update_layout(height=900)
    bump_heatmap['layout']['yaxis']['domain'] = [0.016,1]
else:
    bump_heatmap.update_layout(height=450)
    bump_heatmap['layout']['yaxis']['domain'] = [0.045,1]

st.write('''
         #### Split positions and sector time gaps

         The first chart shows split positions and gaps to the fastest sector times in each of the 5 sectors of the race
         ''')
st.plotly_chart(bump_heatmap)

st.write('''
         #### Individual split and sector time gaps

         Sector plots show the time gaps for the sector only. Split plots show time gaps for the whole course time up to this split (split 1 and sector 1 are identical so no need to have both!)        
         ''')
sector_1, split_1 = st.columns([2, 1])
sector_1_dot_chart = go.Figure(build_dot_chart(race_df, selected_category, 'sector_1'))
sector_1.plotly_chart(sector_1_dot_chart)

sector_2, split_2 = st.columns([2, 1])
sector_2_dot_chart = go.Figure(build_dot_chart(race_df, selected_category, 'sector_2'))
split_2_dot_chart = go.Figure(build_dot_chart(race_df, selected_category, 'split_2'))
sector_2.plotly_chart(sector_2_dot_chart)
split_2.plotly_chart(split_2_dot_chart)

sector_3, split_3 = st.columns([2, 1])
sector_3_dot_chart = go.Figure(build_dot_chart(race_df, selected_category, 'sector_3'))
split_3_dot_chart = go.Figure(build_dot_chart(race_df, selected_category, 'split_3'))
sector_3.plotly_chart(sector_3_dot_chart)
split_3.plotly_chart(split_3_dot_chart)

sector_4, split_4 = st.columns([2, 1])
sector_4_dot_chart = go.Figure(build_dot_chart(race_df, selected_category, 'sector_4'))
split_4_dot_chart = go.Figure(build_dot_chart(race_df, selected_category, 'split_4'))
sector_4.plotly_chart(sector_4_dot_chart)
split_4.plotly_chart(split_4_dot_chart)

sector_5, finish = st.columns([2, 1])
sector_5_dot_chart = go.Figure(build_dot_chart(race_df, selected_category, 'sector_5'))
finish_dot_chart = go.Figure(build_dot_chart(race_df, selected_category, 'finish'))
sector_5.plotly_chart(sector_5_dot_chart)
finish.plotly_chart(finish_dot_chart)

#points_plot = make_season_points_plot(df)
#st.plotly_chart(points_plot)