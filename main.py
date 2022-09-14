import userService as us
import imageService as imageservice
import dbService as dbservice
import about
from io import BytesIO
import streamlit as st
from PIL import Image
from torchvision import transforms
import predictor as predict

st.markdown(
        '<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" '
        'integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">',
        unsafe_allow_html=True)

with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

# container
headerSection = st.container()
registerSection = st.container()
loginSection = st.container()
logOutSection = st.container()
mainSection = st.container()


# def print_result(result):
#     temp = ""
#     for idx, items in enumerate(result):
#         img_src = f'data:image/jpg;base64,{items[1]}'

#         temp += f"""
#             <tr>
#               <th scope="row">{idx + 1}</th>
#               <td><img src="{img_src}"></img></td>
#               <td>{items[2]}</td>
#               <td>{items[3]}</td>
#               <td>{items[4]}</td>
#               <td><button class="btn btn-danger">Delete</button></td>
#             </tr>
#         """
#     return temp

def handle_delete(id):
    dbservice.delete_image(id)
    dbservice.delete_prediction(id)

def show_main_page():
    with mainSection:
        activities = ["Home", "About", "Live Classification", "Upload Image", "Camera Input",  "Prediction Result"]
        choice = st.sidebar.selectbox('Select Activity', activities)
        with st.sidebar:
            st.button("LOGOUT", key="logout", on_click=handle_logout)

        model_name = 'mobilenetv2'
        num_classes = 6
        feature_extract = False
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        idx_to_class = {0: 'Hawar Daun', 1: 'Karat Daun', 2: 'Bukan Daun', 3: 'Daun Sehat', 4: 'bulai',
                        5: 'Bercak Daun'}

        transAug = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std),
             transforms.Resize((256, 256))])
        model_ft, _ = predict.initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

        # Live classification
        if choice == "Live Classification":
            st.title("🎥 Live Classification")
            predict.app_corn_disease_predictor()
            st.markdown(f'''Press Start 👈 to start the show!''')

        # Homepage
        elif choice == 'Home':
            st.markdown('# Aplikasi Demo Klasifikasi Penyakit Jagung dengan Metode CNN')
            st.markdown("""
            ### Tugas Akhir D4 Teknik Informatika Politeknik Harapan Bersama Tegal
            
            Website ini merupakan proyek demonstrasi klasifikasi penyakit tanaman pada daun jagung, yang terdiri dari penyakit bulai, bercak daun, karat daun dan hawar daun.
            
            ## Latar Belakang
            
            Kendala utama yang terjadi pada masa penanaman jagung yaitu gangguan biotis yaitu gangguan oleh makroorganisme (gangguan hama) dan gangguan oleh mikroorganisme (gangguan penyakit). 
            Penyakit tanaman jagung merupakan hasil interaksi dari tiga komponen utama yaitu patogen, inang, dan lingkungan. Kurangnya informasi dan pengetahuan tentang penyakit dari tanaman 
            jagung bagi petani dapat menyebabkan kesalahan diagnosa penyakit yang menyerang tanaman jagung yang berdampak pula pada kesalahan pengendaliannya. Terdapat beberapa penyakit yang 
            dapat menyerang tanaman jagung seperti, penyakit bulai (Downy mildew), hawar daun (Northern Leaf Blight), karat daun (Southern Rust), dan  bercak daun (Southern Leaf Blight). 
            Berdasarkan permasalahan diatas maka perlu dilakukan pendeteksian dini serta pengidentifikasian penyakit tanaman mejadi faktor utama untuk mencegah dan mengurangi penyebaran penyakit pada tanaman jagung. 
            Penyakit yang tidak terdeteksi dan dibiarkan berkembang dapat mengakibatkan kerusakan Oleh karena itu, penelitian ini dilakukan bertujuan untuk mengklasifikasi penyakit jagung. 
            Adapun metode yang digunakan dalam proses klasifikasi adalah Convolutional Neural Network, yang pada prosesnya data training dilatih untuk menghasilkan tingkat akurasi yang maksimal. 
            Convolutional Neural Network (CNN) merupakan salah satu metode Deep learning yang dapat digunakan untuk mendeteksi dan mengenali sebuah objek pada sebuah citra digital. 
            
            ## Fitur Aplikasi
            - Realtime Klasifikasi
            - Klasifikasi menggunakan input dari upload file
            - Klasifikasi menggunakan input kamera
            
            ## Library yang Digunakan
            Pada aplikasi ini menggunakan beberapa library:
            - [Pytorch] - Digunakan sebagai basis deeplearning (pretrained, training dan testing model)
            - [Albumentation] - Augmentasi gambar
            - [Streamlit] - Digunakan sebagai frontend dan juga backend untuk melakukan inferensi model
            - [Streamlit-WebRtc] - Digunakan untuk melakukan inferensi realtime menggunakan WebRtc
            - [Matplotlib] - Library untuk visualisasi data
            [//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)
               [pytorch]: <https://pytorch.org/>
               [albumentation]: <https://albumentations.ai/>
               [Streamlit-WebRtc]: <https://github.com/whitphx/streamlit-webrtc>
               [streamlit]: <https://streamlit.io/>
               [Matplotlib]: <https://matplotlib.org/>
               """)

        # Input Upload
        elif choice == 'Upload Image':
            st.title("🏞 Image Inference")
            image_file = st.file_uploader('Upload Image', type=["jpg", "jpeg"])
            if image_file is not None:
                # insert image into db
                encoded_image = imageservice.encode_image_base64(image_file)
                result_insert_img = dbservice.insert_image(str(encoded_image))

                if result_insert_img != None or result_insert_img != 0:
                    st.success('✅ Image inserted to db.')
                else:
                    st.error('🚨 Failed to insert images.')

                our_static_image = Image.open(image_file)
                idx, output = predict.prediction(image_file, transAug, model_ft)
                output = [float("{:.2f}".format(i)) for i in output]

                # insert prediction to db
                result_insert_prediction = dbservice.insert_prediction(idx_to_class[idx], int(output[idx] * 100), output, result_insert_img)

                if result_insert_prediction != None or result_insert_prediction != 0:
                    st.success('✅ Prediction inserted to db.')
                else:
                    st.error('🚨 Failed to insert prediction.')

                st.image(our_static_image, width=400)
                st.write(f""" ### Prediction: {idx} - {idx_to_class[idx]}""")
                st.write(f"Probability : {int(output[idx] * 100)}%")
                st.write(f"Overall prediction : **{output}**")

        # Input kamera
        elif choice == "Camera Input":
            st.title("📸 Inference using Camera Input")
            img_file_buffer = st.camera_input("Take a picture")
            if img_file_buffer is not None:
                encoded_image = imageservice.encode_image_base64(img_file_buffer)
                # insert image to db
                result_insert_img = dbservice.insert_image(str(encoded_image))

                if result_insert_img != None or result_insert_img != 0:
                    st.success('✅ Image inserted to db.')
                else:
                    st.error('🚨 Failed to insert images.')

                # To read image file buffer as a PIL Image:
                img = Image.open(img_file_buffer)
                idx, output = predict.prediction(img_file_buffer, transAug, model_ft)
                output = [float("{:.2f}".format(i)) for i in output]

                # insert prediction to db
                result_insert_prediction = dbservice.insert_prediction(idx_to_class[idx], int(output[idx] * 100), output,result_insert_img)

                if result_insert_prediction != None or result_insert_prediction != 0:
                    st.success('✅ Prediction inserted to db.')
                else:
                    st.error('🚨 Failed to insert prediction.')

                st.image(img, width=400)
                st.write(f""" ### Prediction: {idx} - {idx_to_class[idx]}""")
                st.write(f"Probability : {int(output[idx] * 100)}%")
                st.write(f"Overall prediction : **{output}**")

        # Dashboard prediksi
        elif choice == "Prediction Result":
            st.title("📈 Prediction Result Dashboard")
            result = dbservice.get_image_and_prediction_all()
            # st.markdown(f'''
            # <table class="table table-striped table-light table-bordered table-hover" style="color:black;">
            #   <caption>List of prediction</caption>
            #   <thead>
            #     <tr>
            #       <th scope="col">No</th>
            #       <th scope="col">Image</th>
            #       <th scope="col">Prediction</th>
            #       <th scope="col">Probability</th>
            #       <th scope="col">Overall Prediction</th>
            #       <th scope="col">Action</th>
            #     </tr>
            #   </thead>
            #   <tbody>
            #     {print_result(result)}
            #   </tbody>
            # </table>''', unsafe_allow_html=True)
            columns = st.columns((1, 3, 2, 2, 2, 1))
            fields = ['No', 'Image', 'Prediction', 'Probability', 'Overall Prediction', 'Action']
            for col, field_name in zip(columns, fields):
                # header
                col.write(field_name)

            for idx, items in enumerate(result):
                decoded_img = imageservice.decode_image_base64(items[1])
                col1, col2, col3, col4, col5, col6 = st.columns((1, 3, 2, 2, 2, 1))
                col1.write(idx)
                col2.image(Image.open(BytesIO(decoded_img)), use_column_width="always")
                col3.write(items[2])
                col4.write(items[3])
                col5.write(items[4])
                col6.button('delete', on_click=handle_delete, args=(items[0], ), key=items[0])

        elif choice == "About":
            about.show_about_page()


# LOGOUT
def handle_logout():
    st.session_state['loggedIn'] = False


def show_logout_page():
    loginSection.empty()
    with logOutSection:
        st.button("Log Out", key="logout", on_click=handle_logout)


# REGISTER
def handle_register(username, password, fullname):
    if us.register(username, password, fullname):
        st.session_state['loggedIn'] = True
        st.success("Success Register")
    else:
        st.error("Register Failed")


def handle_register_session(condition):
    st.session_state['register'] = condition


def show_register_page():
    with registerSection:
        if st.session_state['loggedIn'] == False:
            username = st.text_input(label="Username", value="", placeholder="Enter your Username")
            fullname = st.text_input(label="Fullname", value="", placeholder="Enter your Fullname")
            password = st.text_input(label="Password", value="", placeholder="Enter Password", type="password")
            st.button("Register", on_click=handle_register, args=(username, password, fullname))
            st.button("Login", on_click=handle_register_session, args=(False,))


# LOGIN
def handle_login(username, password):
    if us.login(username, password):
        st.session_state['loggedIn'] = True
    else:
        st.session_state['loggedIn'] = False
        st.error("🚨 Invalid user name or password")


def show_login_page():
    with loginSection:
        st.title("Streamlit Application")
        if st.session_state['loggedIn'] == False:
            username = st.text_input(label="Username", value="", placeholder="Enter your username")
            password = st.text_input(label="Password", value="", placeholder="Enter password", type="password")
            st.button("Login", on_click=handle_login, args=(username, password))
            st.button("Register", on_click=handle_register_session, args=(True,))


if __name__ == '__main__':
    with headerSection:

        # contoh bentu session
        # object = {
        #     "register": False,
        #     "loggedIn": False
        # }

        # first run will have nothing in session_state
        if 'loggedIn' not in st.session_state and 'register' not in st.session_state:
            st.session_state['register'] = False
            st.session_state['loggedIn'] = False
            show_login_page()
        else:
            if st.session_state['loggedIn']:
                show_main_page()
            elif st.session_state['register'] and st.session_state['loggedIn'] == False:
                show_register_page()
            else:
                show_login_page()
