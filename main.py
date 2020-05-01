# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:08:00 2020

@author: bnear
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import os.path

import rasterio
from bokeh.palettes import cividis, gray
from bokeh.layouts import column, row
from bokeh.models import (ColumnDataSource, Toggle, GeoJSONDataSource, HoverTool,
                          Slider, Dropdown, LinearColorMapper, Div, Range1d, LinearAxis, LabelSet, Label)
import pyproj
from bokeh.plotting import figure, curdoc
from shapely.geometry import Point
from shapely import wkt


#%% Sets the sources used in the animation
source_path = os.path.join(os.path.dirname(__file__), 'data')

shapes = pd.read_csv(os.path.join(source_path,'shapefiles.csv'))

shapes['geometry'] = shapes['geometry'].apply(wkt.loads)

hs_92g07w = rasterio.open(os.path.join(source_path,'92g07w_Hillshade_225_utm.tif'))
hs_92g06e = rasterio.open(os.path.join(source_path,'92g06e_Hillshade_225_utm.tif'))
slp_92g07w = rasterio.open(os.path.join(source_path,'slope_07w.tif'))
slp_92g06e = rasterio.open(os.path.join(source_path,'slope_06e.tif'))
rasters = [hs_92g07w, hs_92g06e, slp_92g07w, slp_92g06e ]

gates = pd.read_csv(os.path.join(source_path,'gate_loc.csv'))

fish_locations = pd.read_csv(os.path.join(source_path,'interp_logs_may1.csv'))

river_df = pd.read_csv(os.path.join(source_path,'river_data.csv'))

river_df['Datetime'] = pd.to_datetime(river_df['Datetime'])
fish_locations['Datetime'] = pd.to_datetime(fish_locations['Datetime'])
fish_locations.set_index(['Datetime'], inplace=True)

interp_logs = fish_locations.copy()

mastersheet = pd.read_csv(os.path.join(source_path, 'updated_mastersheet_report.csv'))
mastersheet.set_index(['Tag ID'], inplace=True)

interp_logs['geometry'] = interp_logs['geometry'].apply(wkt.loads)
gates['geometry'] = gates['geometry'].apply(wkt.loads)

fids = ['All','Coho', 'Pink', '130','138', '87-B',  '133-B', '124', '94', '115', '101', '133', '131', '95', '121', '91', '97', '109', '81-B',
         '100', '87', '81', '99', '83', '112', '103', '127', '96', '78', '92', '84', '88-B', '116',
         '82', '154', '159','93', '88', '79', '98', '177', '52', '40', '22', '89', '99-B']
speeds = {'Ludicrous Speed':12,'Fast':6,'Regular': 4, 'Slow': 2, 'Extra Slow': 1}


def make_log_gdf(hourly, log_times, time_ind):
    log_gdf = gpd.GeoDataFrame(hourly.get_group(log_times[time_ind]))
    
    log_gdf.reset_index(inplace=True)
    
    log_gdf['Datetime'] = log_gdf['Datetime'].astype(str)
    
    return log_gdf

class ani():   
        
    logs = interp_logs.copy()
    hourly = logs.groupby(pd.Grouper(freq='H'))
    log_times = list(hourly.groups.keys())
    time_ind = 0
    log_gdf = make_log_gdf(hourly, log_times, time_ind)
    
#%%
    
dash = ani()
fish_stats = Div(text=str('<b>Tag ID: All</br></br>'+ 
                         '<b>Species: Coho & Pink</br></br>'
                         '<b>Sex: Both</br></br>'
                         '<b>Fork Length (cm): 47 to 67</br></br>'
                         '<b>Tag Date: Varied</br></br>'
                         '<b>Tag Site: Varied</br></br>' 
                         '<b>Tag Retrieval: Varied </br></br>'), width=200, style={'font-size': '140%'})
seymour_doc = curdoc()
#%%
###    #Gets the geometry from the shapefiles data and plots them to the input map
###

def plot_background(map_plot):
    
    
    #Plots the rasters
    im1 = rasters[0].read(1)
    im2 = rasters[1].read(1)
    im3 = rasters[2].read(1)
    im4 = rasters[3].read(1)
    
    im1=np.flipud(im1)
    n, m = im1.shape

    im2=np.flipud(im2)
    n2, m2 = im2.shape

    im3=np.flipud(im3)
    im4=np.flipud(im4)
    
    color_mapper = LinearColorMapper(gray(256))
    slope_mapper = LinearColorMapper(cividis(256))

    im_92g07w = map_plot.image(image=[im1], x=499991.2668120749, y=5455186.111482615, dw=(518218.4661728905 - 499991.2668120749), dh=(5483113.492303281 - 5455186.111482615), 
                       color_mapper=color_mapper,alpha=0.3)
        

    im_92g06e = map_plot.image(image=[im2], x=481704.5746754332, y=5455220.297247089, dw=(500022.3013623424 - 481704.5746754332), dh=(5483083.073708355 - 5455220.297247089), 
                       color_mapper=color_mapper,alpha=0.3)
    
    im_92g07w_s = map_plot.image(image=[im3], x=499991.2668120749, y=5455186.111482615, dw=(518218.4661728905 - 499991.2668120749), dh=(5483113.492303281 - 5455186.111482615), 
                   color_mapper=slope_mapper,alpha=0.3)
        

    im_92g06e_s = map_plot.image(image=[im4], x=481704.5746754332, y=5455220.297247089, dw=(500022.3013623424 - 481704.5746754332), dh=(5483083.073708355 - 5455220.297247089), 
                       color_mapper=slope_mapper,alpha=0.3)
    
    
    
    
    
    #Plots Seymour River and Lynn Creek
    sey_lynn_shape = shapes[(shapes['CREEK_NAME'] == 'SEYMOUR RIVER') | (shapes['CREEK_NAME'] == 'LYNN CREEK')]
    
    sey_lynn_shape = gpd.GeoDataFrame(sey_lynn_shape)
    
    sey_lynn_source = GeoJSONDataSource(geojson = sey_lynn_shape.to_json())
    

    seymour_lines = map_plot.multi_line('xs','ys', source = sey_lynn_source,
                                        line_color = 'blue', line_width = 3, line_alpha=0.5)
    
    #Plots the rest of the streams much thinner
    stream_shape = shapes[(shapes['shapefile'] == 'Streams') 
                      & (~((shapes['CREEK_NAME'] == 'SEYMOUR RIVER') |(shapes['CREEK_NAME'] == 'LYNN CREEK')))]
    
    stream_shape = gpd.GeoDataFrame(stream_shape)
    
    stream_source = GeoJSONDataSource(geojson = stream_shape.to_json())

    steam_lines = map_plot.multi_line('xs','ys', source = stream_source, 
                                      line_color='blue', line_width = 1, line_alpha=0.5)
    
    #Plots the polygon shapes for Seymour and Lynn Rivers
    river_shape = shapes[shapes['shapefile'] == 'Rivers']
    
    river_shape = gpd.GeoDataFrame(river_shape)
    
    river_source  = GeoJSONDataSource(geojson = river_shape.to_json())
 
    river_poly = map_plot.patches('xs','ys', source = river_source,
                                  fill_color='blue', line_color='blue',
                                  fill_alpha=0.5, line_width = 0.2)
    
    
    #Plots the ocean and coastline
    ocean_shape = shapes[shapes['shapefile']=='Ocean']
    
    ocean_shape = gpd.GeoDataFrame(ocean_shape)
    
    ocean_source  = GeoJSONDataSource(geojson = ocean_shape.to_json())
                                  
    ocean_poly = map_plot.patches('xs','ys', source = ocean_source,
                                  fill_color='blue', line_color='blue',
                                  fill_alpha=0.8, line_width = 0.2)
    
    #Plots the lakes
    lakes_shape = shapes[shapes['shapefile']=='Lakes']
    
    lakes_shape = gpd.GeoDataFrame(lakes_shape)
    
    lakes_source  = GeoJSONDataSource(geojson = lakes_shape.to_json())
                                  
    lakes_poly = map_plot.patches('xs','ys', source = lakes_source,
                                  fill_color='blue', line_color='blue',
                                  fill_alpha=0.8, line_width = 0.2)    
    gate_gdf = gpd.GeoDataFrame(gates)
    
    gate_gsrc = GeoJSONDataSource(geojson = gate_gdf.to_json())
    
    gates_glyphs = map_plot.diamond_cross(x='x', y='y', size='size', color='color', source=gate_gsrc, fill_alpha=0)
    
    labels = LabelSet(x='x', y='y', text='Location', level='glyph',
              x_offset='x_offset', y_offset=-10, source=gate_gsrc, render_mode='canvas', text_font_style = 'bold', text_alpha=1)
    
    map_plot.add_layout(labels)


    return map_plot



#%%
def set_dash_logs(new_fid, interp_logs):
    if new_fid == 'All':
        dash.logs = interp_logs.copy()      
        
    elif new_fid == 'Coho':
        dash.logs = interp_logs[interp_logs['Species'] == 'Coho'].copy()

    elif new_fid == 'Pink':
        dash.logs = interp_logs[interp_logs['Species'] == 'Pink'].copy()
        
    else:
        dash.logs = interp_logs[interp_logs['FID'] == new_fid].copy()
        
    dash.hourly = dash.logs.groupby(pd.Grouper(freq='H'))
    dash.log_times = list(dash.hourly.groups.keys())
    dash.time_ind = 0
    dash.log_gdf = make_log_gdf(dash.hourly, dash.log_times, dash.time_ind)
    
    
#%%
### Sets the logs GeoJSONDataSource
###
toggle = Toggle(label = "Play", button_type = "success",width = 450)
speed_dropdown = Dropdown(label='Regular', button_type="warning", value='Regular', menu=['Ludicrous Speed','Fast','Regular', 'Slow', 'Extra Slow'], width=150)
#%%
def anime_update(n):
    time_ind = slider.value+n
    if time_ind > (len(dash.hourly.groups) - 1):
        time_ind = 0
    slider.value = int(time_ind)

def tog_update():
    if toggle.active:
        anime_update(speeds[speed_dropdown.value])
        toggle.label='Pause'
    else:
        toggle.label='Play'


seymour_doc.add_periodic_callback(tog_update, 100)



def data_update(attrname, old, new): 
    #Gets the slider value
    time_ind = slider.value
    
    flow_point_src.data.update(ColumnDataSource(fl_grps.get_group(dash.log_times[time_ind])).data)    
    
    log_gsrc.geojson = make_log_gdf(dash.hourly, dash.log_times, time_ind).to_json()
    
    slider.title = str('Date: ' + str(dash.log_times[time_ind]))
    
def fish_dropdown_update(attrname, old, new):
    #Updates the interpolated points
    fid = fish_dropdown.value
    
    set_dash_logs(fid, interp_logs)
    
    flow_point_src.data.update(ColumnDataSource(fl_grps.get_group(dash.log_times[0])).data)    
    
    log_gsrc.geojson = make_log_gdf(dash.hourly, dash.log_times, 0).to_json()
    
    slider.value = 0
    
    slider.start = 0
    
    slider.end = len(dash.hourly.groups)
    
    slider.value = 0
    
    slider.title = str(dash.log_times[0])
    
    fish_dropdown.label = fid
    
    if fid == 'All':
        fish_stats.text=str('<b>Tag ID: All</br></br>'+ 
                         '<b>Species: Coho & Pink</br></br>'
                         '<b>Sex: Both</br></br>'
                         '<b>Fork Length (cm): 47 to 67</br></br>'
                         '<b>Tag Date: Varied</br></br>'
                         '<b>Tag Site: Varied</br></br>' 
                         '<b>Tag Retrieval: Varied </br></br>')
    elif fid == 'Coho':
        fish_stats.text=str('<b>Tag ID: All</br></br>'+ 
                         '<b>Species: Coho</br></br>'
                         '<b>Sex: Both</br></br>'
                         '<b>Fork Length (cm): 47 to 67</br></br>'
                         '<b>Tag Date: Varied</br></br>'
                         '<b>Tag Site: Varied</br></br>' 
                         '<b>Tag Retrieval: Varied </br></br>')
    elif fid == 'Pink':
        fish_stats.text=str('<b>Tag ID: All</br></br>'+ 
                         '<b>Species: Pink</br></br>'
                         '<b>Sex: Both</br></br>'
                         '<b>Fork Length (cm): 47 to 56</br></br>'
                         '<b>Tag Date: Varied</br></br>'
                         '<b>Tag Site: Varied</br></br>' 
                         '<b>Tag Retrieval: Varied </br></br>')
    else: 
        fish_stats.text =str('<b>Tag ID:<b> ' + fid + '</br></br>'+ 
                         '<b>Species:<b> ' + str(mastersheet.loc[fid]['Species']) + '</br></br>'
                         '<b>Sex:<b> ' + str(mastersheet.loc[fid]['Sex']) + '</br></br>'
                         '<b>Fork Length (cm):<b> ' + str(mastersheet.loc[fid]['Fork Length (cm)']) + '</br></br>'
                         '<b>Tag Date:<b> ' + str(mastersheet.loc[fid]['Tag Date']) + '</br></br>'
                         '<b>Tag Site:<b> ' +  str(mastersheet.loc[fid]['Tag Site']) + '</br></br>' 
                         '<b>Tag Retrieval:<b> ' + str(mastersheet.loc[fid]['Tag Retrieval Date']) + '</br></br>')

def speed_dropdown_update(attrname, old, new):
    #Updates the callback period
    new_speed = speed_dropdown.value
    
    speed_dropdown.label = new_speed
    
    slider.step = speeds[new_speed]
    
        

#%%
flow_plot = figure(plot_width = 900, plot_height = 400, 
          title = 'River Flows', y_range = (0,100),
          x_axis_label = 'Time', y_axis_label = 'FLOW m^3/s',
          tools='pan,box_zoom,wheel_zoom,reset')

flow_plot.extra_y_ranges = {'Temperature': Range1d(start=0, end =20)}
flow_plot.add_layout(LinearAxis(y_range_name='Temperature', axis_label = 'TEMPERATURE C'), 'right')

flow_df = river_df[['Datetime','Flow','Temperature']]

flow_df.loc['Datetime'] = pd.to_datetime(flow_df['Datetime'])

#Plots the River flow rate
flow_src = ColumnDataSource(flow_df)

flow_logs = flow_plot.line(x='Datetime', y='Flow', source = flow_src, line_color='black', line_width = 1)

temp_logs = flow_plot.line(x='Datetime', y='Temperature', y_range_name= 'Temperature', source = flow_src, line_color='green', line_width = 1, alpha=0.7)
 
fl_grps = flow_df.groupby('Datetime')
#Plots the moving circle on flow rates
flow_point_src= ColumnDataSource(fl_grps.get_group(ani.log_times[0]))

river_timepoint = flow_plot.circle(x='Datetime', y='Flow', source = flow_point_src, color='red', alpha=0.7, size=10)
    
#%%
p = figure(plot_height=900, plot_width=600, 
           x_range=[493666.6666, 505000], y_range=[5460000, 5477000],
           tools='pan,wheel_zoom,reset')

p2 = figure(plot_height=500, plot_width=900, 
           x_range=[494200, 505000], y_range=[5460000, 5466000],
           tools='pan,wheel_zoom,reset')


plot_background(p)
plot_background(p2)

set_dash_logs('All',interp_logs)

log_gsrc = GeoJSONDataSource(geojson = dash.log_gdf.to_json())

slider = Slider(start = 0, end = len(dash.hourly.groups),
                value = 0, title=str('Date: ' + str(dash.log_times[0])) ,step =1)

slider.on_change('value',data_update)

speed_list = list(speeds.keys())
fish_dropdown = Dropdown(label='All', button_type="warning", value='All', menu=fids, width=200)

fish_dropdown.on_change('value', fish_dropdown_update)

speed_dropdown.on_change('value', speed_dropdown_update)

flocations = p.circle(x='x', y='y', size='size', color='color', alpha='alpha', source=log_gsrc)

flocations_p2 = p2.circle(x='x', y='y', size='size', color='color', alpha='alpha', source=log_gsrc)


hover_tool = HoverTool(tooltips=[
            ('Fish', '@FID'),('Species', '@Species'), ('Sex', '@Sex'), ('Fork Length (cm)', '@Fork_Length_cm')
            ], renderers=[flocations])

hover_tool2 = HoverTool(tooltips=[
            ('Fish', '@FID'),('Species', '@Species'), ('Sex', '@Sex'), ('Fork Length (cm)', '@Fork_Length_cm'),
            ], renderers=[flocations_p2])

hover_tool_flow = HoverTool(tooltips=[
            ('Flow', '@Flow'), ('Temperature', '@Temperature'),
            ], renderers=[river_timepoint])


p.tools.append(hover_tool)
p2.tools.append(hover_tool2)
flow_plot.tools.append(hover_tool_flow)

seymour_doc.add_root(row(column(slider,p2, flow_plot),column(row(toggle, speed_dropdown),p),column(fish_dropdown, fish_stats)))


