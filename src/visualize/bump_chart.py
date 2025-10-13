import pandas as pd
import plotly.graph_objects as go

def add_district_traces(
                        fig: go.Figure, 
                        df_income: pd.DataFrame, 
                        custom_colors: list, 
                        year_order: list
                        ):
    '''
    Adds traces for each district to the bump chart, including lines and annotations.
    '''
    # Add a line for each district
    for i, district in enumerate(df_income['District Name'].unique()):
        district_data = df_income[df_income['District Name'] == district]
        
        # Get the last point for the annotation
        last_point = district_data.iloc[-1]

        # Get the index of the last year in the year_order
        last_year_index = year_order.index(last_point['Year'])

        # Get the color from the custom color list (using the index 'i')
        line_color = custom_colors[i % len(custom_colors)]  

        # Add the line trace
        # fig.add_trace(
        #     go.Scatter(
        #         x=district_data['Year'],
        #         y=district_data['Ranking'],
        #         mode='lines+markers',
        #         name=district,
        #         line=dict(color=line_color),  
        #         marker=dict(size=20), 
        #         hovertemplate='<br><b>Year: </b>%{x}<br><b>Ranking: </b>%{y}<br><b>Income: </b>%{customdata[0]:,.2f}<extra></extra>',  # Include 'Income' in hover
        #         customdata=district_data[['Income']],  # Add 'Income' to custom data for hover
        #     )
        # )
        fig.add_trace(
            go.Scatter(
                x=district_data["Year"],
                y=district_data["Ranking"],
                mode="lines+markers",
                name=district,
                line=dict(color=line_color),
                marker=dict(size=8),  # ★ ここでマーカー小さく
                hovertemplate="<br><b>Year: </b>%{x}"
                              "<br><b>Ranking: </b>%{y}"
                              "<br><b>Income: </b>%{customdata[0]:,.2f}"
                              "<extra></extra>",
                customdata=district_data[["Income"]],
            )
        )
        # Add annotation with a buffer to the left of the last year point
        buffer = 0.05  # Adjust this value to increase or decrease the buffer
        fig.add_annotation(
            x=last_year_index + buffer,  # Add buffer to the x position
            y=last_point['Ranking'],
            text=district,
            showarrow=False,  # Remove the arrow
            font=dict(size=15, color=line_color),  # Match the color of the label with the line
            xanchor='left',  # Align text to the left of the specified x position
        )

def add_subtitle(fig, subtitle, subtitle_font_size=14, subtitle_color='gray', y_offset=0.92, x_offset=0.5):
    '''
    Adds a subtitle to a Plotly figure.
    '''
    fig.add_annotation(
        text=subtitle,
        x=x_offset,  # Horizontal position
        y=y_offset,  # Vertical position
        xref='paper',
        yref='paper',
        showarrow=False,
        font=dict(size=subtitle_font_size, color=subtitle_color),
        align='center',
    )
    return fig

def add_footer(fig, footer, footer_font_size=12, footer_color='gray', y_offset=-0.1, x_offset=0.5):
    '''
    Adds a footer to a Plotly figure.
    '''
    fig.add_annotation(
        text=footer,
        x=x_offset,  # Horizontal position
        y=y_offset,  # Vertical position 
        xref='paper',
        yref='paper',
        showarrow=False,
        font=dict(size=footer_font_size, color=footer_color),
        align='center',
    )
    return fig

def customize_layout(fig, year_order=None):
    '''
    Customizes the layout of the figure, including titles, axis settings, and grid visibility.
    '''
    # Invert the y-axis because higher rankings should appear at the top
    fig.update_yaxes(
        autorange='reversed',
        tickmode='linear',
        dtick=1,
        showticklabels=False  # Remove y-axis labels
    )
    # x軸をカテゴリ順に固定（←重要）
    if year_order is not None:
        fig.update_xaxes(type="category", categoryorder="array", 
                         categoryarray=year_order, tickangle=270)

    # Customize layout settings (titles, grid, etc.)
    fig.update_layout(
        # title={
        # 'text': 'Ranking of Valencia Districts by Income: 2015 vs. 2022',
        # 'y': 0.925,  # Adjust vertical position (default is 1.0)
        # },
        title=None,
        plot_bgcolor='rgba(255,255,255,1)',
        paper_bgcolor='rgba(255,255,255,1)',
        font=dict(family='Poppins'),
        height=800, 
        width=800,   
        showlegend=False,  # Remove the legend
        xaxis_showgrid=False,  # Remove grid lines from the x-axis
        yaxis_showgrid=False,  # Remove grid lines from the y-axis
    )
def add_ranking_annotations(fig, df_income, year_order):
    '''
    Adds annotations for the ranking at each marker on the bump chart.
    '''
    for district in df_income['District Name'].unique():
        district_data = df_income[df_income['District Name'] == district]
        # for _, row in district_data.iterrows():
            # Get the position of the year in the year order
            # year_index = year_order.index(row['Year'])
            # y_offset = 0.45
            # fig.add_annotation(
            #     x=year_index,
            #     y=row['Ranking'] + y_offset,
            #     text=str(int(row['Ranking'])),  # Display the ranking as an integer
            #     showarrow=False,
            #     font=dict(size=12, color='white'),  # Customize font size and color
            #     xanchor='center',  # Align text to the marker
            #     yanchor='bottom',  # Position the text just below the marker
            # )
def get_custom_colors(background='light'):
    '''
    Returns a list of custom colors in hex format for the bump chart lines.
    Based on the background type (light or dark).
    Parameters:
        background (str): Type of background, either 'light' or 'dark'.
    Returns:
        list: A list of custom colors for the specified background type.
    '''
    # Colors for light background
    colors_for_light_bg = [
        '#4B7CCC',  # Medium Blue
        '#F2668B',  # Pink
        '#03A688',  # Medium Teal
        '#FFAE3E',  # Amber
        '#B782B8',  # Purple
        '#A67F63',  # Chestnut
        '#0E8B92',  # Deep Teal
        '#D4AC2C',  # Bronze Yellow
        '#7E9F5C',  # Forest Green
        '#F7BCA3',  # Burnt Orange
        '#E63946',  # Red
        '#7DA9A7',  # Slate Blue
        '#457B9D',  # Steel Blue
        '#E094AC',  # Rose
        '#1D3557',  # Dark Blue
        '#2A9D8F',  # Teal Green
        '#B38A44',  # Antique Gold
        '#C68045',  # Copper
        '#264653',  # Charcoal Blue
    ]

    # Colors for dark background
    colors_for_dark_bg = [
        '#A8C9F4',  # Pastel Blue
        '#FFB3C6',  # Pastel Pink
        '#A0E5D6',  # Soft Teal
        '#FFEC88',  # Pastel Yellow
        '#E2A8D3',  # Pastel Lavender
        '#D0B89D',  # Soft Tan
        '#80D0D4',  # Light Aqua
        '#F1E59B',  # Pale Lemon
        '#B4D79E',  # Light Mint Green
        '#F2B48C',  # Soft Coral
        '#FFB4B4',  # Light Red
        '#A2D1D1',  # Light Cyan
        '#7FBCD1',  # Light Steel Blue
        '#F1A8D6',  # Light Rose
        '#5C7F9E',  # Light Denim Blue
        '#70C7B7',  # Soft Green-Teal
        '#C8A67F',  # Pastel Gold
        '#D7A584',  # Light Copper
        '#A4B6D4',  # Soft Charcoal Blue
    ]

    if background == 'dark':
        return colors_for_dark_bg
    else:
        return colors_for_light_bg
def create_bump_chart(df_income):
    '''
    Creates and displays the bump chart using the provided dataframe.
    '''
    # Get the list of custom colors
    custom_colors = get_custom_colors()  

    # Create the bump chart figure
    fig = go.Figure()

    # Add district traces to the figure
    year_order = sorted(df_income['Year'].unique()) 
    add_district_traces(fig, df_income, custom_colors, year_order)

    # Add a subtitle to the figure
    subtitle = 'Visualizing changes in income across Valencia\'s districts over seven years'
    add_subtitle(fig, subtitle, subtitle_font_size=15, subtitle_color='grey', y_offset=1.050, x_offset=-0.0875)

    # Add a footer to the figure
    footer = 'Source: INE (Instituto Nacional de Estadística). Data retrieved on November 2024.'
    add_footer(fig, footer, footer_font_size=12, footer_color='grey', y_offset=-0.1, x_offset=0.35)

    # Customize the layout of the figure
    customize_layout(fig)

    # Show the figure
    fig.show()