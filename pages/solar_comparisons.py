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
    def pvgis_pvcalc_response(lat, long, peakpower, loss, angle):
        pvcalc = requests.get(
            f'https://re.jrc.ec.europa.eu/api/v5_2/PVcalc?'
            f'lat={lat}&'
            f'lon={long}&'
            f'peakpower={peakpower}&'
            f'loss={loss}&'
            f'angle={angle}&'
            f'outputformat=json&'
            f'browser=0')

        pvcalc_json = pvcalc.json()

        pvcalc_outputs = pvcalc_json['outputs']

        pvcalc_months = pvcalc_outputs['monthly']

        ghi_monthly = []

        for key, value in pvcalc_months.items():
            month = value[0]
            ghi = month['E_m']
            ghi_monthly.append(ghi)

        ghi_df = pd.DataFrame({
            'Average Monthly GHI (kWh/m\u00B2)': ghi_monthly,
            'Month': ["January", "February", "March",
                      "April", "May", "June",
                      "July", "August", "September",
                      "October", "November", "December"]
        })

        return ghi_df

    @st.cache(allow_output_mutation=True)
    def dnv_json_response(lat, long, id, acc,
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

        dnv_json_resp = dnv_post.json()

        monthly_result = dnv_json_resp['SummaryMonthly']
        monthly_ghi = monthly_result['GlobHor']

        system_attributes_df = pd.DataFrame.from_dict(json_normalize(dnv_json_resp['SystemAttributes']),
                                                      orient='columns')
        metadata_df = pd.DataFrame.from_dict(json_normalize(dnv_json_resp['Metadata']), orient='columns')

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

        summary_annually_df = pd.DataFrame.from_dict(json_normalize(dnv_json_resp['SummaryAnnual']),
                                                     orient='columns')

        summary_monthly_df = pd.DataFrame.from_dict(json_normalize(dnv_json_resp['SummaryMonthly']),
                                                    orient='columns')

        hourly_energy_df = pd.DataFrame.from_dict(json_normalize(dnv_json_resp['Hourly']),
                                                  orient='columns')

        return system_attributes_df, metadata_df, ghi_chart, ghi_df, summary_annually_df, summary_monthly_df, hourly_energy_df

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

    dnv_column, pvgis_col = st.columns(2)

    with dnv_column:
        # design for the app
        address_option = st.selectbox('Would you like to get coordinates from an address?', ('Yes', 'No'))

        if address_option == 'Yes':
            locator = Nominatim(user_agent="MyGeocoder")
            # 1 - conveneint function to delay between geocoding calls
            geocode = RateLimiter(locator.geocode, min_delay_seconds=1)
            # 2- - create location column
            address = st.text_input('Please enter a address to get location.',
                                    help="Example: 18450 Bernardo Trails Dr San Diego California")
            location = locator.geocode(address)
            if location:
                # st.write("Latitude = {}, Longitude = {}".format(location.latitude, location.longitude))
                Latitude = location.latitude
                Longitude = location.longitude
        else:
            # Streamlit user input values
            Latitude = st.text_input('Please enter your latitude')
            Longitude = st.text_input('Please enter your longitude')

        if Latitude and Longitude:
            poi = folium.Marker(location=[Latitude, Longitude], popup="Your selected point of interest")
            poi.add_to(m)
            bounding_box = poi.get_bounds()
            m.fit_bounds([bounding_box])
            folium_static(m, width=800, height=800)

        if address_option and Latitude and Longitude:
            ProjRefID = st.text_input('Please input the name for this project')
            ACCapacity = st.text_input('Insert the AC Capacity (kWAC)',
                                       help="Project AC capacity (kWac). Default = 1")
            DCCapacity = st.text_input('Insert the DC Capacity (kWDC)',
                                       help="Project DC capacity (kWdc). Default = ["
                                            "same as ACCapacity]")
            Mounting = st.selectbox('What sort of racking would you like?', ('ground_fixed', 'ground_1axis'),
                                    help="Type of PV panel mounting: ground_fixed, ground_1axis")
            Azimuth = st.text_input('Input the Azimuth',
                                    help="Azimuth angle orientation of arrays. N=0, E = 90, S = 180, W = 270 (or -90). Default = [180 for Northern hemisphere, 0 for Southern hemisphere]")
            Tilt = st.text_input("Insert the tilt angle",
                                 help="Tilt angle of PV modules. For 1-axis trackers, use the maximum rotation angle. Default = [estimated based on selected Mounting type]")
            ModuleConfig = st.selectbox('What Module Config would you like?', ('mod1P', 'mod2P'),
                                        help="Module layout on racking reflecting Landscape or Portrait orientation: mod1P, mod2P")
            Pitch = st.text_input("Insert the Pitch",
                                  help="Distance between centerline axis of each row of panels (meters)")

        dnv_run = st.button("Get DNV Estimates")

        if dnv_run:
            dnv_values = dnv_json_response(Latitude, Longitude, ProjRefID, ACCapacity,
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
                hourly_df = dnv_values[6]

                st.dataframe(energy_df)

    with pvgis_col:
        # design for the app
        address_option = st.selectbox('Would you like to get coordinates from an address?', ('Yes', 'No'))

        if address_option == 'Yes':
            locator = Nominatim(user_agent="MyGeocoder")
            # 1 - conveneint function to delay between geocoding calls
            geocode = RateLimiter(locator.geocode, min_delay_seconds=1)
            # 2- - create location column
            address = st.text_input('Please enter a address to get location.',
                                    help="Example: 18450 Bernardo Trails Dr San Diego California")
            location = locator.geocode(address)
            if location:
                # st.write("Latitude = {}, Longitude = {}".format(location.latitude, location.longitude))
                Latitude = location.latitude
                Longitude = location.longitude
        else:
            # Streamlit user input values
            Latitude = st.text_input('Please enter your latitude')
            Longitude = st.text_input('Please enter your longitude')

        if Latitude and Longitude:
            poi = folium.Marker(location=[Latitude, Longitude], popup="Your selected point of interest")
            poi.add_to(m)
            bounding_box = poi.get_bounds()
            m.fit_bounds([bounding_box])
            folium_static(m, width=800, height=800)

        if address_option and Latitude and Longitude:
            pvgis_peakpower = st.text_input('Please enter the peak power for the site',
                                            help="Nominal power of the PV system, in kW.")
            pvgis_loss = st.text_input('Please enter the estimated system losses',
                                       help="Sum of system losses, in percent. The estimated system losses are all the losses in the system, "
                                            "which cause the power actually delivered to the electricity grid to be lower than the power"
                                            " produced by the PV modules")
            pvgis_slope = st.text_input('Please enter the slope',
                                        help="Inclination angle from horizontal plane of the (fixed) PV system. "
                                             "This is the angle of the PV modules from the horizontal plane, "
                                             "for a fixed (non-tracking) mounting. Default = 0")

        pvgis_run = st.button("Get PVGIS Estimates")

        if pvgis_run:
            pvcalc_df = pvgis_pvcalc_response(Latitude, Longitude, pvgis_peakpower, pvgis_loss, pvgis_slope)

            st.dataframe(pvcalc_df)


