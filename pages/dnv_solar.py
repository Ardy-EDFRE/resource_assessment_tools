import altair as alt
import folium
from streamlit_folium import folium_static
from geopy.extra.rate_limiter import RateLimiter
from geopy import Nominatim
import io
import pandas as pd
from pandas import json_normalize
import requests
import streamlit as st


def app():
    st.title("DNV Solar Resource Compass API ")

    st.markdown("**How it works**")
    st.markdown(f"""
                Solar Resource Compass accesses and compares irradiance data from multiple data providers
                and allows you to see how they compare for your project location. By default,
                Solar Resource Compass will access data from NREL (satellite and MTS datasets), 
                SolarGIS, Meteonorm and DNV’s SunSpot irradiance model. The results of the analysis
                include a statistical comparison of the available sources presented in a convenient
                table, chart and map. User can also upload their own data for comparsion.
                
                Solar Resource Compass also generates monthly soiling  loss estimates for both dust soiling
                and snow soiling. By incorporating industry standard models and DNV analytics, precipitation
                and snowfall data is automatically accessed and used to estimate the impact on energy generation.
                The loss profiles are presented in a monthly table to use in popular energy modeling software.
                
                With the proliferation of bifacial modules, developers and investors also need guidance
                on albedo - the light reflected off the ground and on to the back surface of solar modules.
                Solar Resource Compass addresses this need through the use of a proprietary model that calculates
                a monthly albedo profile that can be used in any commerical energy modeling software.
                """)

    @st.cache(allow_output_mutation=True)
    def get_chart(data, field, title, x_value, y_value, x_title, y_title):
        hover = alt.selection_single(
            fields=[field],
            nearest=True,
            on="mouseover",
            empty="none",
        )

        line = (
            alt.Chart(data, title=title)
                .mark_line()
                .encode(
                x=x_value,
                y=y_value,
            )
        )

        # Draw points on the line, and highlight based on selection
        points = line.transform_filter(hover).mark_circle(size=65)

        # Draw a rule at the location of the selection
        tooltips = (
            alt.Chart(data)
                .mark_rule()
                .encode(
                x=x_value,
                y=y_value,
                opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
                tooltip=[
                    alt.Tooltip(x_value, title=x_title),
                    alt.Tooltip(y_value, title=y_title),
                ],
            )
                .add_selection(hover)
        )
        return (line + points + tooltips).interactive()

    @st.cache(allow_output_mutation=True)
    def json_response(lat, long, id, acc,
                      dcc, mount, tilt, azi,
                      mtech, msize, mconfig,
                      isbi, invtype, land, mwash,
                      clear, pitch, dcd, avail):

        dnv_post = requests.get(f'https://api.src.dnv.com/api/site/energy?'
                                f'Latitude={lat}'
                                f'&Longitude={long}'
                                f'&ProjRefID={id}'
                                f'&ACCapacity={acc}'
                                f'&DCCapacity={dcc}'
                                f'&Mounting={mount}'
                                f'&Tilt={tilt}'
                                f'&Azimuth={azi}'
                                f'&ModuleTech={mtech}'
                                f'&ModuleSize={msize}'
                                f'&ModuleConfig={mconfig}'
                                f'&IsBifacial={isbi}'
                                f'&InverterType={invtype}'
                                f'&LandUse={land}'
                                f'&ManualWash={mwash}'
                                f'&Clearance={clear}'
                                f'&Pitch={pitch}'
                                f'&DCDerate={dcd}'
                                f'&Availability={avail}', headers={'X-ApiKey': src_key},
                                verify=False)

        dnv_json_response = dnv_post.json()

        monthly_result = dnv_json_response['SummaryMonthly']
        monthly_ghi = monthly_result['GlobHor']

        system_attributes_df = pd.DataFrame.from_dict(json_normalize(dnv_json_response['SystemAttributes']),
                                                      orient='columns')
        metadata_df = pd.DataFrame.from_dict(json_normalize(dnv_json_response['Metadata']), orient='columns')

        ghi_x = []

        for key, value in monthly_ghi.items():
            ghi_x.append(value)

        ghi_df = pd.DataFrame({
            'Average Monthly GHI (kWh/m\u00B2)': ghi_x,
            'Month': ["January", "February", "March",
                      "April", "May", "June",
                      "July", "August", "September",
                      "October", "November", "December"]
        })
        ghi_chart = get_chart(ghi_df,
                              "Average Monthly GHI (kWh/m\u00B2)",
                              "Average Monthly GHI",
                              "Month",
                              "Average Monthly GHI (kWh/m\u00B2)",
                              "Month",
                              "GHI (kWh/m\u00B2)")

        summary_annually_df = pd.DataFrame.from_dict(json_normalize(dnv_json_response['SummaryAnnual']),
                                                     orient='columns')

        summary_monthly_df = pd.DataFrame.from_dict(json_normalize(dnv_json_response['SummaryMonthly']),
                                                    orient='columns')

        data_values = [system_attributes_df, metadata_df, ghi_chart, ghi_df, summary_annually_df, summary_monthly_df]

        return system_attributes_df, metadata_df, ghi_chart, ghi_df, summary_annually_df, summary_monthly_df

    @st.cache(allow_output_mutation=True)
    def write_excel(sys_df, meta_df, eng_df, sum_an_df, sum_m_df):
        with io.BytesIO() as buffer:
            writer = pd.ExcelWriter(buffer)
            sys_df.to_excel(writer, sheet_name="SystemAttributes", index=False)
            meta_df.to_excel(writer, sheet_name="Metadata", index=False)
            eng_df.to_excel(writer, sheet_name="NetEnergyMonthly", index=False)
            sum_an_df.to_excel(writer, sheet_name="SummaryAnnually", index=False)
            sum_m_df.to_excel(writer, sheet_name="SummaryMonthly", index=False)
            writer.save()

            csv = buffer.getvalue()

            return csv

    @st.cache(allow_output_mutation=True)
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    # Hard coded values for DNV
    Latitude = 0.0
    Longitude = 0.0
    ProjRefID = ""
    ACCapacity = 0.0
    DCCapacity = 0.0
    Mounting = ""
    Azimuth = 0.0
    Tilt = 0.0
    ModuleConfig = ""
    Pitch = ""
    ModuleTech = "mono"
    ModuleSize = "2266x1134x35"
    IsBifacial = "true"
    InverterType = "string"
    LandUse = ""
    ManualWash = ""
    Clearance = 1
    # Pitch (can there be a formula added to the interface to derive from GCR %? If they select Ground Fixed Tilt and a
    # specific degree of tilt then it would be the plane of array x cos(25 deg tilt for example)
    DCDerate = ""
    Availability = ""
    src_key = 'MFobC0FXHVNVZCVES0xsUjFFWSlZYBdIVWkwNnk6dWM='

    # showing the maps
    m = folium.Map(tiles='OpenStreetMap')

    # Add custom base maps to folium
    base_maps = {
        'Google Maps': folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
            attr='Google',
            name='Google Maps',
            overlay=True,
            control=True
        ),
        'Google Satellite': folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr='Google',
            name='Google Satellite',
            overlay=True,
            control=True
        ),
        'Google Terrain': folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
            attr='Google',
            name='Google Terrain',
            overlay=True,
            control=True
        ),
        'Google Satellite Hybrid': folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
            attr='Google',
            name='Google Satellite',
            overlay=True,
            control=True
        ),
        'Esri Satellite': folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Esri Satellite',
            overlay=True,
            control=True
        )
    }

    # Add custom basemaps
    base_maps['Google Maps'].add_to(m)
    base_maps['Google Satellite Hybrid'].add_to(m)

    # design for the app
    address_option = st.sidebar.selectbox('Would you like to get coordinates from an address?', ('Yes', 'No'))

    if address_option == 'Yes':
        locator = Nominatim(user_agent="MyGeocoder")
        # 1 - conveneint function to delay between geocoding calls
        geocode = RateLimiter(locator.geocode, min_delay_seconds=1)
        # 2- - create location column
        address = st.sidebar.text_input('Please enter a address to get location.', help="Example: 18450 Bernardo Trails Dr San Diego California")
        location = locator.geocode(address)
        if location:
            # st.write("Latitude = {}, Longitude = {}".format(location.latitude, location.longitude))
            Latitude = location.latitude
            Longitude = location.longitude
    else:
        # Streamlit user input values
        Latitude = st.sidebar.text_input('Please enter your latitude')
        Longitude = st.sidebar.text_input('Please enter your longitude')

    if Latitude and Longitude:
        poi = folium.Marker(location=[Latitude, Longitude], popup="Your selected point of interest")
        poi.add_to(m)
        bounding_box = poi.get_bounds()
        m.fit_bounds([bounding_box])
        folium_static(m, width=800, height=800)

    if address_option and Latitude and Longitude:
        ProjRefID = st.sidebar.text_input('Please input the name for this project')
        ACCapacity = st.sidebar.text_input('Insert the AC Capacity (kWAC)', help="Project AC capacity (kWac). Default = 1")
        DCCapacity = st.sidebar.text_input('Insert the DC Capacity (kWDC)', help="Project DC capacity (kWdc). Default = ["
                                                                                 "same as ACCapacity]")
        Mounting = st.sidebar.selectbox('What sort of racking would you like?', ('ground_fixed', 'ground_1axis'), help="Type of PV panel mounting: ground_fixed, ground_1axis")
        Azimuth = st.sidebar.text_input('Input the Azimuth', help="Azimuth angle orientation of arrays. N=0, E = 90, S = 180, W = 270 (or -90). Default = [180 for Northern hemisphere, 0 for Southern hemisphere]")
        Tilt = st.sidebar.text_input("Insert the tilt angle", help="Tilt angle of PV modules. For 1-axis trackers, use the maximum rotation angle. Default = [estimated based on selected Mounting type]")
        ModuleConfig = st.sidebar.selectbox('What Module Config would you like?', ('mod1P', 'mod2P'), help="Module layout on racking reflecting Landscape or Portrait orientation: mod1P, mod2P")
        Pitch = st.sidebar.text_input("Insert the Pitch", help="Distance between centerline axis of each row of panels (meters)")

    RunDnv = st.sidebar.button("Submit for DNV run")

    if RunDnv:
        dnv_values = json_response(Latitude, Longitude, ProjRefID, ACCapacity,
                                   DCCapacity, Mounting, Tilt, Azimuth,
                                   ModuleTech, ModuleSize, ModuleConfig,
                                   IsBifacial, InverterType, LandUse, ManualWash,
                                   Clearance, Pitch, DCDerate, Availability)

        if dnv_values:
            system_attr_df = dnv_values[0]
            metadata_df = dnv_values[1]
            energy_chart = dnv_values[2]
            energy_df = dnv_values[3]
            sum_annual_df = dnv_values[4]
            sum_monthly_df = dnv_values[5]

            excel = write_excel(system_attr_df, metadata_df,
                                energy_df, sum_annual_df,
                                sum_monthly_df)

            st.subheader('System Attribute Inputs')
            st.dataframe(system_attr_df)
            st.subheader('Metadata')
            st.dataframe(metadata_df)
            st.subheader("Monthly Net Energy")
            st.altair_chart(energy_chart.interactive(), use_container_width=True)
            st.dataframe(energy_df)
            st.subheader('Summary result annually')
            st.dataframe(sum_annual_df)
            st.subheader('Summary result monthly')
            st.dataframe(sum_monthly_df)

            st.sidebar.download_button(
                label="Download",
                data=excel,
                file_name="EnergyEstimate.xlsx",
                mime="application/vnd.ms-excel"
            )

    st.markdown(
        "![Watermark Company Logo](https://raw.githubusercontent.com/Ardy-EDFRE/resource_assessment_tools/main/edf_small_logo.png)")

        # Pages to add on maybe
        # page_names_to_funcs = {
        #     "—": about_page,
        #     "Energy Estimate Page": main_page,
        #     "Download Energy Estimate": download_page
        # }
        #
        # selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
        # page_names_to_funcs[selected_page]()
        #
        #
        # dnv_post = requests.get(f'https://api.src.dnv.com/api/site/energy?'
            #                         f'Latitude={Latitude}'
            #                         f'&Longitude={Longitude}'
            #                         f'&ProjRefID={ProjRefID}'
            #                         f'&ACCapacity={ACCapacity}'
            #                         f'&DCCapacity={DCCapacity}'
            #                         f'&Mounting={Mounting}'
            #                         f'&Tilt={Tilt}'
            #                         f'&Azimuth={Azimuth}'
            #                         f'&ModuleTech={ModuleTech}'
            #                         f'&ModuleSize={ModuleSize}'
            #                         f'&ModuleConfig={ModuleConfig}'
            #                         f'&IsBifacial={IsBifacial}'
            #                         f'&InverterType={InverterType}'
            #                         f'&LandUse={LandUse}'
            #                         f'&ManualWash={ManualWash}'
            #                         f'&Clearance={Clearance}'
            #                         f'&Pitch={Pitch}'
            #                         f'&DCDerate={DCDerate}'
            #                         f'&Availability={Availability}', headers={'X-ApiKey': src_key},
            #                         verify=False)
            #
            # dnv_json = dnv_post.json()
            #
            # system_attributes = dnv_json['SystemAttributes']
            # metadata = dnv_json['Metadata']
            # monthly_result = dnv_json['SummaryMonthly']
            # summary_annually = dnv_json['SummaryAnnual']
            # loss_tree_annual = dnv_json['LossTreeAnnual']
            # monthly_ghi = monthly_result['GlobHor']

            # monthly_albedo = monthly_result['Albedo']
            # monthly_snowfall = monthly_result['Snowfall']
            # monthly_precipitation = monthly_result['Precip']
            # monthly_soiling_loss = monthly_result['SoilingLoss']
            # monthly_incident = monthly_result['GlobInc']
            # monthly_temp = monthly_result['T_Amb']
            # monthly_wind_vel = monthly_result['WindVel']
            # monthly_egrid = monthly_result['E_Grid']
            # #
            # # incident_x = []
            # #
            # # for key, value in monthly_incident.items():
            # #     incident_x.append(value)
            # #
            # # incident_df = pd.DataFrame({
            # #     'Average Monthly Global Incident (kWh/m\u00B2)': incident_x,
            # #     'Month': ["January", "February", "March",
            # #               "April", "May", "June",
            # #               "July", "August", "September",
            # #               "October", "November", "December"]
            # # })
            # #
            # # incident_chart = get_chart(incident_df,
            # #                            "Average Monthly Global Incident (kWh/m\u00B2)",
            # #                            "Average Monthly Global Incident",
            # #                            "Month",
            # #                            "Average Monthly Global Incident (kWh/m\u00B2)",
            # #                            "Month",
            # #                            "Global Incident (kWh/m\u00B2)")
            # #
            # # velocity_x = []
            # #
            # # for key, value in monthly_wind_vel.items():
            # #     velocity_x.append(value)
            # #
            # # velocity_df = pd.DataFrame({
            # #     'Average Monthly Wind Velocity (m/s)': velocity_x,
            # #     'Month': ["January", "February", "March",
            # #               "April", "May", "June",
            # #               "July", "August", "September",
            # #               "October", "November", "December"]
            # # })
            # #
            # # velocity_chart = get_chart(velocity_df,
            # #                            "Average Monthly Wind Velocity (m/s)",
            # #                            "Average Monthly Wind Velocity",
            # #                            "Month",
            # #                            "Average Monthly Wind Velocity (m/s)",
            # #                            "Month",
            # #                            "Wind Velocity (m/s)")
            # #
            # # egrid_x = []
            # #
            # # for key, value in monthly_egrid.items():
            # #     egrid_x.append(value)
            # #
            # # egrid_df = pd.DataFrame({
            # #     'Average Monthly Energy Injected to Grid (kVAh)': egrid_x,
            # #     'Month': ["January", "February", "March",
            # #               "April", "May", "June",
            # #               "July", "August", "September",
            # #               "October", "November", "December"]
            # # })
            # #
            # # egrid_chart = get_chart(egrid_df,
            # #                         "Average Monthly Energy Injected to Grid (kVAh)",
            # #                         "Average Monthly Energy Injected to Grid",
            # #                         "Month",
            # #                         "Average Monthly Energy Injected to Grid (kVAh)",
            # #                         "Month",
            # #                         "Energy Injected to Grid (kVAh)")
            # #
            # # st.altair_chart(incident_chart.interactive(), use_container_width=True)
            # # st.altair_chart(velocity_chart.interactive(), use_container_width=True)
            # # st.altair_chart(egrid_chart.interactive(), use_container_width=True)
            #
            # st.subheader('System Attributes used for energy estimate run.')
            # system_attributes_df = pd.DataFrame.from_dict(json_normalize(dnv_json['SystemAttributes']), orient='columns')
            # st.dataframe(system_attributes_df)
            # sys_attr_csv = convert_df(system_attributes_df)
            #
            # st.subheader('Metadata')
            # metadata_df = pd.DataFrame.from_dict(json_normalize(dnv_json['Metadata']), orient='columns')
            # st.dataframe(metadata_df)
            # metadata_csv = convert_df(metadata_df)
            #
            # st.subheader("Monthly Net Energy")
            # st.altair_chart(ghi_chart.interactive(), use_container_width=True)
            # st.dataframe(ghi_df)
            # ghi_csv = convert_df(ghi_df)
            #
            # st.subheader('Summary result annually')
            # summary_annually_df = pd.DataFrame.from_dict(json_normalize(dnv_json['SummaryAnnual']), orient='columns')
            # st.dataframe(summary_annually_df)
            # sum_annual_csv = convert_df(summary_annually_df)
            #
            # st.subheader('Summary result monthly')
            # summary_monthly_df = pd.DataFrame.from_dict(json_normalize(dnv_json['SummaryMonthly']), orient='columns')
            # st.dataframe(summary_monthly_df)
            # sum_monthly_csv = convert_df(summary_monthly_df)