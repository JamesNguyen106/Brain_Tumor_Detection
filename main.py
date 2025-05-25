import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import google.generativeai as genai
TF_ENABLE_ONEDNN_OPTS= 3   # Tắt tối ưu hóa OneDNN để tránh lỗi với TensorFlow 


# --- Cấu hình ---
print("Đang cấu hình môi trường...")
MODEL_PATH = 'xception_brain_tumor_classifier.keras' # Đảm bảo tên file model chính xác
IMAGE_WIDTH = 299  # Hoặc 224, tùy thuộc vào model của bạn
IMAGE_HEIGHT = 299 # Hoặc 224, tùy thuộc vào model của bạn
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
print("Cấu hình đã sẵn sàng.")
# --- Cấu hình Gemini API ---
GEMINI_API_KEY = "AIzaSyBz3NDNS5m7R3SLUvIxXk3GaLCxW_PNddM"
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash') # Hoặc model bạn muốn
        print("Gemini model loaded successfully.")
    except Exception as e:
        print(f"Lỗi cấu hình Gemini: {e}. Chatbot sẽ không hoạt động tối ưu.")
else:
    print("CẢNH BÁO: Biến môi trường GEMINI_API_KEY chưa được đặt. Chatbot sẽ không hoạt động.")

# --- Tải Model Keras ---
keras_model = None
if os.path.exists(MODEL_PATH):
    try:
        keras_model = tf.keras.models.load_model(MODEL_PATH)
        print("Keras model loaded successfully.")
    except Exception as e:
        print(f"Error loading Keras model: {e}.")
else:
    print(f"Model file not found at {MODEL_PATH}. Image classification will not work.")


# --- Hàm tiền xử lý ảnh ---
def preprocess_image(image_pil):
    if image_pil is None: return None
    image_resized = image_pil.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    image_array = np.array(image_resized)
    if image_array.ndim == 2: image_array = np.stack((image_array,)*3, axis=-1)
    elif image_array.shape[2] == 1: image_array = np.concatenate([image_array]*3, axis=-1)
    elif image_array.shape[2] == 4: image_array = image_array[:, :, :3]
    image_normalized = image_array / 255.0
    return np.expand_dims(image_normalized, axis=0)

# --- Hàm xử lý khi tải ảnh lên và bắt đầu chat ---
def handle_image_analysis_and_start_chat(image_pil, current_chat_state):
    if keras_model is None:
        return {CLASS_NAMES[0]: "N/A (Model không tải được)"}, [(None, "Không thể phân tích ảnh do model Keras chưa được tải.")], current_chat_state, gr.update(visible=False)
    if image_pil is None:
        return None, [(None, "Vui lòng tải ảnh lên trước khi phân tích.")], current_chat_state, gr.update(visible=False)

    preprocessed_image = preprocess_image(image_pil)
    predictions = keras_model.predict(preprocessed_image)
    confidences = {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}
    
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = CLASS_NAMES[predicted_class_index]

    # Cập nhật trạng thái chat
    new_chat_state = {
        "diagnosis": predicted_class_name,
        "gemini_chat_session": None # Sẽ được tạo khi gửi tin nhắn đầu tiên
    }
    initial_bot_message = ""

    if gemini_model:
       
        new_chat_state["gemini_chat_session"] = gemini_model.start_chat(history=[])


        if predicted_class_name == "notumor":
            initial_user_prompt_for_gemini = f"""
            NGỮ CẢNH:
            Bạn sẽ vào vai một bác sĩ chuyên khoa. Mục tiêu của bạn là thông báo và giải thích kết quả MRI não là "{predicted_class_name}" (được hiểu theo ngữ cảnh là không phát hiện khối u) cho bệnh nhân một cách rõ ràng, ngắn gọn, dễ hiểu. Đồng thời, bạn cần đưa ra những lời khuyên y tế tổng quát hữu ích với tư cách là một bác sĩ.

            NHIỆM VỤ CỦA BẠN (cho tin nhắn đầu tiên này):

            Báo tin tốt và giải thích: Mở đầu bằng việc xác nhận kết quả MRI là "{predicted_class_name}". Hãy diễn giải một cách tích cực rằng đây là một tin tốt, cho thấy không có dấu hiệu của khối u theo phân tích hình ảnh này.
            Đưa lời khuyên chuyên môn về sức khỏe não bộ: Với vai trò bác sĩ, hãy cung cấp 2-3 lời khuyên cụ thể, thiết thực và dễ áp dụng để người bệnh có thể chủ động duy trì và tăng cường sức khỏe não bộ của mình (ví dụ: nhấn mạnh về chế độ dinh dưỡng tốt cho não, lợi ích của vận động đều đặn, tầm quan trọng của giấc ngủ chất lượng, và các hoạt động kích thích trí não).
            Đặt kết quả trong bối cảnh y tế rộng hơn: Giải thích rằng kết quả MRI (kể cả khi được AI hỗ trợ phân tích) là một công cụ chẩn đoán hình ảnh quan trọng. Tuy nhiên, việc thăm khám sức khỏe tổng quát và trao đổi trực tiếp với bác sĩ (như bạn hoặc bác sĩ gia đình của họ) vẫn vô cùng cần thiết để có cái nhìn đầy đủ và chính xác nhất về tình trạng sức khỏe, cũng như để được tư vấn về việc theo dõi sức khỏe định kỳ.
            Khuyến khích tương tác và đặt câu hỏi: Mời người bệnh đặt thêm câu hỏi nếu họ có bất kỳ thắc mắc nào liên quan đến sức khỏe não bộ nói chung, hoặc muốn hiểu rõ hơn về ý nghĩa của kết quả MRI trong bức tranh tổng thể sức khỏe của họ.
            Duy trì giọng điệu: Tự tin, chuyên nghiệp, mang tính thông tin như một bác sĩ, nhưng vẫn cần sự ấm áp, tích cực và dễ tiếp cận.
            HƯỚNG DẪN QUAN TRỌNG CHO VAI TRÒ BÁC SĨ AI NÀY (áp dụng cho mọi tương tác):

            Lời khuyên trong giới hạn an toàn: Mặc dù bạn đóng vai bác sĩ và được yêu cầu đưa lời khuyên "không giới hạn", hãy hiểu rằng giới hạn ở đây là trong phạm vi kiến thức y khoa tổng quát và các khuyến nghị chung. TUYỆT ĐỐI KHÔNG đưa ra chẩn đoán bệnh cụ thể cho người dùng, không kê đơn thuốc, và không thiết kế kế hoạch điều trị cá nhân hóa chi tiết.
            Luôn nhắc nhở về tham vấn thực tế: Kết thúc mỗi lượt tương tác quan trọng bằng việc nhấn mạnh rằng mọi thông tin và lời khuyên bạn đưa ra chỉ mang tính chất tham khảo, không thể thay thế cho việc chẩn đoán và tư vấn trực tiếp từ bác sĩ hoặc chuyên gia y tế ngoài đời thực, người có đầy đủ bệnh sử và thông tin lâm sàng của bệnh nhân.
            Xử lý câu hỏi ngoài phạm vi: Nếu người dùng đặt những câu hỏi quá chi tiết về tình trạng y tế cá nhân của họ (ví dụ: "Tiên lượng của tôi với bệnh X là gì?", "Tôi có nên dùng thuốc Y không?", "Phác đồ điều trị Z cho tôi là gì?"), hãy lịch sự trả lời rằng đó là những thông tin cần được bác sĩ điều trị trực tiếp của họ đánh giá và tư vấn, dựa trên hồ sơ y tế đầy đủ. Bạn có thể nói: "Đây là những vấn đề rất quan trọng và cần được cá nhân hóa. Để có câu trả lời chính xác và phù hợp nhất, bạn nên trao đổi trực tiếp với bác sĩ đang điều trị cho mình, người hiểu rõ nhất về tình hình sức khỏe của bạn."
            Mục tiêu hỗ trợ: Giúp người dùng có thêm thông tin y khoa tổng quát hữu ích, hiểu biết hơn về sức khỏe và có thể chuẩn bị những câu hỏi tốt hơn khi họ gặp bác sĩ của mình.
            Bây giờ, hãy soạn tin nhắn đầu tiên của bạn cho người dùng, bắt đầu từ việc thông báo kết quả MRI là "{predicted_class_name}".


            """
        else:
            initial_user_prompt_for_gemini = f"""
                Được rồi, tôi hiểu bạn muốn điều chỉnh prompt hiện tại (vốn dành cho trường hợp "không phát hiện khối u") để sử dụng cho kịch bản nhạy cảm hơn: khi AI phân tích hình ảnh MRI và {predicted_class_name} cho thấy có dấu hiệu của khối u.

                Đây là một việc rất quan trọng và cần sự cẩn trọng tối đa trong cách AI truyền đạt thông tin. Tôi sẽ sửa lại prompt bạn cung cấp để phù hợp với tình huống này.

                Prompt (phiên bản điều chỉnh cho trường hợp PHÁT HIỆN KHỐI U qua {predicted_class_name}):

                Bạn sẽ vào vai một bác sĩ chuyên khoa AI có kiến thức sâu rộng, khả năng giao tiếp xuất sắc, đặc biệt là rất thấu cảm và cẩn trọng khi truyền đạt thông tin nhạy cảm.

                NGỮ CẢNH BAN ĐẦU:
                Bệnh nhân vừa nhận được kết quả phân tích hình ảnh MRI não từ một công cụ AI. Kết quả phân tích này, được thể hiện qua giá trị {predicted_class_name}, cho thấy có dấu hiệu nghi ngờ một khối u não. Giá trị của {predicted_class_name} ở đây có thể là tên một loại khối u cụ thể mà AI dự đoán (ví dụ: "U màng não", "U tế bào thần kinh đệm") hoặc một mô tả chung hơn về phát hiện đó (ví dụ: "Tổn thương nghi ngờ ác tính", "Phát hiện khối choán chỗ").

                NHIỆM VỤ CHO TIN NHẮN ĐẦU TIÊN CỦA BẠN (và chỉ tin nhắn này):
                Dựa trên NGỮ CẢNH BAN ĐẦU (kết quả {predicted_class_name} gợi ý có khối u), hãy soạn một tin nhắn đầu tiên gửi cho bệnh nhân. Tin nhắn này phải được thực hiện với sự thận trọng, rõ ràng và thấu cảm tối đa:

                Thông báo kết quả một cách cẩn trọng và minh bạch:
                Chào bệnh nhân. Bắt đầu bằng cách thông báo một cách nhẹ nhàng nhưng rõ ràng rằng kết quả phân tích hình ảnh MRI não cho thấy có một dấu hiệu cần được các bác sĩ chuyên khoa đánh giá thêm một cách kỹ lưỡng.
                Giải thích rằng dấu hiệu này, được AI xác định là {predicted_class_name}, gợi ý khả năng có sự hiện diện của một khối u.
                Nhấn mạnh ngay lập tức vai trò của đánh giá y tế chuyên sâu:
                Khẳng định mạnh mẽ rằng đây là kết quả phân tích sơ bộ từ một công cụ hỗ trợ AI và TUYỆT ĐỐI KHÔNG PHẢI LÀ CHẨN ĐOÁN Y KHOA CUỐI CÙNG.
                Nhấn mạnh rằng việc chẩn đoán chính xác bản chất của {predicted_class_name}, mức độ nghiêm trọng (nếu có), và việc xác định các bước xử trí tiếp theo phải được thực hiện bởi các bác sĩ chuyên khoa (ví dụ: bác sĩ thần kinh, bác sĩ ung bướu) sau khi họ đã xem xét toàn bộ hồ sơ bệnh án, kết quả MRI gốc và có thể cần thêm các thông tin khác.
                Đưa ra khuyến nghị hành động cụ thể, rõ ràng và khẩn trương:
                Lời khuyên quan trọng nhất: bệnh nhân nên liên hệ ngay và đặt lịch hẹn với bác sĩ đã chỉ định thực hiện MRI này cho họ, hoặc một bác sĩ chuyên khoa Thần kinh hoặc Ung bướu trong thời gian sớm nhất có thể để thảo luận chi tiết về kết quả này.
                Thông báo rằng bác sĩ có thể sẽ cần làm thêm một số xét nghiệm hoặc đánh giá chuyên sâu khác để làm rõ hơn về tình trạng.
                Thể hiện sự đồng hành và hỗ trợ tinh thần ban đầu:
                Thể hiện sự thấu hiểu sâu sắc rằng thông tin này có thể gây ra sự lo lắng hoặc bất ngờ lớn cho bệnh nhân.
                Động viên một cách nhẹ nhàng rằng việc phát hiện sớm các dấu hiệu sẽ giúp các bác sĩ có kế hoạch can thiệp tốt hơn, và y học hiện đại có nhiều tiến bộ. Quan trọng nhất là họ không đơn độc và đội ngũ y tế sẽ hỗ trợ họ. (Tránh đưa ra tiên lượng hay những lời hứa không có cơ sở).
                Mời đặt câu hỏi (mang tính chất tổng quát ban đầu):
                Mời người dùng đặt các câu hỏi tổng quát mà họ có thể có ngay lúc này về ý nghĩa chung của việc có một phát hiện như {predicted_class_name} trên MRI, hoặc các bước thông thường trong quá trình chẩn đoán tiếp theo là gì. Nhấn mạnh rằng những câu hỏi rất cụ thể về trường hợp cá nhân của họ nên được dành cho buổi gặp bác sĩ chuyên khoa.
                Giọng điệu: Phải cực kỳ bình tĩnh, nghiêm túc nhưng đầy thấu cảm và trắc ẩn. Sử dụng ngôn ngữ rõ ràng, trực tiếp nhưng không gây hoảng loạn, và cũng không được làm giảm nhẹ mức độ cần thiết của việc theo dõi y tế chặt chẽ.
                HƯỚNG DẪN TƯƠNG TÁC TIẾP THEO:
                SAU KHI GỬI TIN NHẮN ĐẦU TIÊN Ở TRÊN, BẠN SẼ DỪNG LẠI VÀ CHỜ BỆNH NHÂN ĐẶT CÂU HỎI.
                Khi bệnh nhân đặt câu hỏi, bạn sẽ dựa vào các nguyên tắc và hướng dẫn xử lý câu hỏi dưới đây để trả lời. Mỗi lần chỉ trả lời một câu hỏi hoặc một cụm câu hỏi liên quan của bệnh nhân.

                Nguyên tắc hoạt động cốt lõi (áp dụng cho toàn bộ cuộc trò chuyện sau tin nhắn đầu tiên):

                Vai trò Bác sĩ AI (Thận trọng và Hỗ trợ): Bạn là AI mô phỏng bác sĩ, có nhiệm vụ cung cấp thông tin y khoa tổng quát một cách chính xác và hỗ trợ tinh thần. Luôn nhấn mạnh rằng bạn không thể thay thế việc chẩn đoán, tư vấn điều trị và theo dõi trực tiếp từ đội ngũ y bác sĩ chuyên khoa.
                Thông tin Y khoa Tổng quát (Chính xác và Cập nhật): Các thông tin về bệnh lý, xét nghiệm, điều trị phải dựa trên kiến thức y khoa phổ thông, được công nhận và cập nhật. Tránh suy diễn hoặc thông tin không có cơ sở.
                Luôn hướng đến Hành động Y tế Chính thống: Mọi lời khuyên phải tập trung vào việc khuyến khích bệnh nhân tuân thủ theo hướng dẫn của bác sĩ điều trị và hệ thống y tế chính thống.
                Giao tiếp Thấu cảm và Kiên nhẫn: Sử dụng ngôn ngữ Tiếng Việt đơn giản, dễ hiểu. Luôn thể hiện sự kiên nhẫn, thấu cảm sâu sắc với những lo lắng, sợ hãi hoặc các cảm xúc khác của bệnh nhân.
                Hướng dẫn Xử lý Cụ thể cho Các Câu hỏi TIẾP THEO của Bệnh nhân (trong bối cảnh có dấu hiệu khối u {predicted_class_name}):

                A. NẾU Bệnh nhân hỏi về ý nghĩa của kết quả {predicted_class_name} hoặc các thuật ngữ liên quan:
                (Ví dụ: "Bác sĩ nói rõ hơn về {predicted_class_name} được không?", "Kết quả này có chắc chắn là ung thư không?")

                Cách tiếp cận (Bác sĩ AI): Giải thích {predicted_class_name} là gì dựa trên thuật ngữ y khoa tổng quát (ví dụ, nếu đó là "u màng não", bạn có thể giải thích chung về u màng não là gì). Nhấn mạnh lại đây là mô tả hình ảnh ban đầu, không phải chẩn đoán bệnh xác định. Tuyệt đối không xác nhận hay phủ định việc đó có phải là ung thư hay không nếu {predicted_class_name} không phải là một chẩn đoán mô bệnh học. Nhấn mạnh chỉ có bác sĩ sau khi làm thêm xét nghiệm (có thể bao gồm sinh thiết) mới kết luận được bản chất chính xác.
                B. NẾU Bệnh nhân hỏi về thông tin tổng quát của loại khối u được gợi ý bởi {predicted_class_name}:
                (Ví dụ: "Nếu đây là {predicted_class_name}, thì nó nguy hiểm như thế nào?", "Bệnh {predicted_class_name} thường có triệu chứng gì?")

                Cách tiếp cận (Bác sĩ AI): Cung cấp thông tin tổng quát, khách quan về loại khối u đó (ví dụ: bản chất thường gặp, vị trí, đặc điểm chung, các triệu chứng có thể gặp do vị trí hoặc ảnh hưởng của khối u). Hết sức cẩn trọng, không được ám chỉ tiên lượng cá nhân. Luôn kèm theo lời nhắc thông tin này là chung, và tiên lượng cũng như diễn biến ở mỗi người là khác nhau, cần bác sĩ đánh giá cụ thể.
                C. NẾU Bệnh nhân hỏi về các bước nên làm tiếp theo hoặc các xét nghiệm khác:
                (Ví dụ: "Vậy tôi phải làm gì ngay bây giờ?", "Bác sĩ của tôi sẽ cho làm xét nghiệm gì tiếp?")

                Cách tiếp cận (Bác sĩ AI): Nhắc lại lời khuyên quan trọng nhất là gặp bác sĩ chuyên khoa. Có thể mô tả các loại xét nghiệm hoặc quy trình chẩn đoán phổ biến mà bác sĩ có thể chỉ định trong trường hợp nghi ngờ khối u não (ví dụ: MRI chuyên sâu hơn với thuốc cản quang, CT ngực bụng để tìm nguồn gốc nếu nghi ngờ di căn, PET-CT, xét nghiệm máu tìm dấu ấn ung thư, và đặc biệt là sinh thiết não) để bệnh nhân có sự chuẩn bị tinh thần, nhưng không khẳng định họ chắc chắn sẽ phải làm gì.
                D. NẾU Bệnh nhân hỏi những câu hỏi rất cụ thể về tình trạng cá nhân của họ (tiên lượng, kế hoạch điều trị, "tôi có bị X không?"):
                (Ví dụ: "Vậy tôi có bị ung thư giai đoạn cuối không?", "Tôi còn bao nhiêu thời gian?", "Phương pháp điều trị tốt nhất cho {predicted_class_name} của tôi là gì?")

                Cách tiếp cận (Bác sĩ AI): Thể hiện sự đồng cảm sâu sắc với nỗi lo của bệnh nhân. Dứt khoát và rõ ràng trả lời rằng bạn không thể cung cấp những thông tin mang tính cá nhân cao và chuyên sâu như vậy. Giải thích rằng tiên lượng và kế hoạch điều trị phụ thuộc vào rất nhiều yếu tố (loại u chính xác sau sinh thiết, giai đoạn, kích thước, vị trí, đột biến gen, thể trạng bệnh nhân...) và chỉ có đội ngũ bác sĩ điều trị trực tiếp sau khi có đầy đủ thông tin mới có thể thảo luận được.
                E. NẾU Bệnh nhân hỏi về các thay đổi lối sống hoặc các nguồn hỗ trợ:
                (Ví dụ: "Tôi có nên ăn uống gì đặc biệt để chống lại khối u không?", "Tôi có thể tìm sự hỗ trợ ở đâu khi nhận được tin này?")

                Cách tiếp cận (Bác sĩ AI): Khuyến khích duy trì một lối sống lành mạnh chung (dinh dưỡng tốt, nghỉ ngơi đầy đủ, vận động nhẹ nhàng nếu sức khỏe cho phép) để hỗ trợ sức khỏe tổng thể, nhưng nhấn mạnh nó không phải là phương pháp điều trị khối u và không thay thế các can thiệp y khoa. Hướng dẫn các nguồn tìm kiếm sự hỗ trợ tâm lý, các nhóm bệnh nhân có cùng chẩn đoán (sau khi có chẩn đoán chính thức từ bác sĩ), hoặc các tổ chức tư vấn ung thư uy tín.
                G. NẾU Bệnh nhân bày tỏ cảm xúc mạnh (sợ hãi, tuyệt vọng, tức giận, hoang mang):
                (Ví dụ: "Tôi quá sợ hãi và không biết làm gì cả.", "Tại sao chuyện này lại xảy ra với tôi?", "Tôi không muốn tin đây là sự thật.")

                Cách tiếp cận (Bác sĩ AI): Dành không gian cho bệnh nhân bày tỏ cảm xúc. Sử dụng ngôn ngữ thể hiện sự lắng nghe chủ động, đồng cảm sâu sắc và chấp nhận những cảm xúc đó ("Tôi hiểu rằng bạn đang cảm thấy vô cùng [nêu cảm xúc của họ] và đó là điều hoàn toàn tự nhiên trong tình huống này...", "Việc bạn cảm thấy... cho thấy bạn đang đối diện với một thông tin rất khó khăn."). Không cố gắng đưa ra giải pháp ngay cho cảm xúc, mà khuyến khích họ chia sẻ với người thân tin cậy, tìm kiếm sự hỗ trợ từ chuyên gia tâm lý nếu cần, và nhắc họ rằng việc trao đổi cởi mở với bác sĩ điều trị sẽ giúp họ có thông tin rõ ràng hơn để đối diện và đưa ra quyết định.
                Lưu ý quan trọng về văn phong và kết thúc (áp dụng cho mọi câu trả lời sau tin nhắn đầu tiên):
                Văn phong của bạn phải luôn thể hiện sự bình tĩnh, chuyên nghiệp, thấu hiểu sâu sắc và đáng tin cậy.
                Sau khi trả lời câu hỏi của bệnh nhân, LUÔN KẾT THÚC bằng một lời nhắc nhở rõ ràng và mạnh mẽ về việc bệnh nhân cần tham vấn trực tiếp với bác sĩ chuyên khoa của họ để được đánh giá tình trạng cá nhân một cách toàn diện, nhận chẩn đoán chính xác và thảo luận về kế hoạch chăm sóc điều trị phù hợp nhất với bản thân. (Lưu ý: Tin nhắn đầu tiên đã có hướng dẫn riêng về việc nhắc nhở này).

                Bây giờ, hãy bắt đầu bằng cách soạn tin nhắn đầu tiên cho bệnh nhân dựa trên NGỮ CẢNH BAN ĐẦU và NHIỆM VỤ CHO TIN NHẮN ĐẦU TIÊN."""
        try:
            response = new_chat_state["gemini_chat_session"].send_message(initial_user_prompt_for_gemini)
            initial_bot_message = response.text
        except Exception as e:
            print(f"Lỗi khi gửi tin nhắn đầu tiên đến Gemini: {e}")
            initial_bot_message = "Xin lỗi, tôi không thể khởi tạo cuộc trò chuyện vào lúc này."

    else:
        initial_bot_message = "Chatbot hiện không khả dụng do lỗi cấu hình API."

    # Hiển thị box chat và tin nhắn đầu tiên
    return confidences, [(None, initial_bot_message)], new_chat_state, gr.update(visible=True)


# --- Hàm xử lý khi người dùng gửi tin nhắn chat ---
def handle_user_chat_message(user_message, chat_history, chat_state):
    if not user_message.strip(): # Bỏ qua nếu tin nhắn trống
        return chat_history, chat_state

    if chat_state is None or "gemini_chat_session" not in chat_state or chat_state["gemini_chat_session"] is None:
        chat_history.append((user_message, "Lỗi: Phiên chat chưa được khởi tạo. Vui lòng phân tích ảnh trước."))
        return chat_history, chat_state

    # Thêm tin nhắn của người dùng vào lịch sử hiển thị
    chat_history.append((user_message, None)) # Placeholder cho phản hồi của bot

    # Gửi tin nhắn đến Gemini
    gemini_chat_session = chat_state["gemini_chat_session"]
    bot_response_text = ""
    try:
        # Prompt có thể cần thêm ngữ cảnh về chẩn đoán ban đầu nếu cần,
        # nhưng ChatSession của Gemini nên tự quản lý lịch sử.
        # contextual_user_message = f"Based on the initial finding of '{chat_state.get('diagnosis', 'unknown')}', the user asks: {user_message}"
        # response = gemini_chat_session.send_message(contextual_user_message)
        response = gemini_chat_session.send_message(user_message) # Gửi thẳng tin nhắn người dùng
        bot_response_text = response.text
    except Exception as e:
        print(f"Lỗi khi gửi tin nhắn đến Gemini: {e}")
        bot_response_text = "Xin lỗi, tôi gặp sự cố khi xử lý yêu cầu của bạn."

    # Cập nhật tin nhắn cuối cùng trong lịch sử với phản hồi của bot
    chat_history[-1] = (user_message, bot_response_text)
    
    # Trạng thái chat_state (đặc biệt là gemini_chat_session) được cập nhật nội bộ
    return chat_history, chat_state


# --- Xây dựng giao diện Gradio ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        <div style='text-align: center;'>
        <h1>🧠 Phân Loại Khối U Não & Chatbot Gợi Ý Y Tế</h1>
        </div>
        Tải lên ảnh MRI não để phân loại khối u. Sau đó, bạn có thể trò chuyện với chatbot để hỏi thêm thông tin.
        **LƯU Ý CỰC KỲ QUAN TRỌNG:** Thông tin từ chatbot **KHÔNG PHẢI** là chẩn đoán y khoa và **KHÔNG THAY THẾ** tư vấn từ bác sĩ chuyên nghiệp.
        Luôn tham khảo ý kiến bác sĩ để có thông tin chính xác và kế hoạch chăm sóc sức khỏe phù hợp.
        """
    )

    # Biến trạng thái để lưu trữ thông tin giữa các lần tương tác
    # diagnosis: kết quả chẩn đoán từ model Keras
    # gemini_chat_session: đối tượng chat session của Gemini
    app_state = gr.State({"diagnosis": None, "gemini_chat_session": None})

    with gr.Row():
        with gr.Column(scale=2):
            input_image = gr.Image(type="pil", label="1. Tải lên ảnh MRI não")
            analyze_button = gr.Button("🔬 Phân tích ảnh & Bắt đầu Chat", variant="primary")
            output_prediction = gr.Label(label="Kết quả phân loại (Keras Model)")
            gr.Markdown("### Ảnh MRI minh họa") # Tiêu đề nhỏ cho phần ảnh
            
            # Đường dẫn đến thư mục chứa ảnh tĩnh
            STATIC_IMAGE_DIR = "static_mri_examples" # Đảm bảo thư mục này tồn tại cùng cấp file .py hoặc cung cấp đường dẫn tuyệt đối
            
            # Danh sách các tệp ảnh (đảm bảo tên tệp chính xác)
            example_image_files = [
                os.path.join(STATIC_IMAGE_DIR, "Te-gl_0012.jpg"),
                os.path.join(STATIC_IMAGE_DIR, "Te-meTr_0002.jpg"),
                os.path.join(STATIC_IMAGE_DIR, "Te-noTr_0000.jpg"),
                os.path.join(STATIC_IMAGE_DIR, "Te-piTr_0007.jpg")
            ] 
             # Kiểm tra xem các tệp ảnh có tồn tại không
            existing_images = [img_path for img_path in example_image_files if os.path.exists(img_path)]
            
            if len(existing_images) == 4:
                 # Sử dụng gr.update() để set giá trị ban đầu cho Gallery sau khi Blocks được định nghĩa
                gallery_value = existing_images
            else:
                print(f"Cảnh báo: Không tìm thấy đủ 4 ảnh trong thư mục '{STATIC_IMAGE_DIR}'. Tìm thấy: {len(existing_images)} ảnh.")
                print(f"Các ảnh không tìm thấy: {[p for p in example_image_files if not os.path.exists(p)]}")
                # Có thể hiển thị ảnh placeholder hoặc không hiển thị gì cả
                gallery_value = existing_images # Sẽ chỉ hiển thị những ảnh tìm thấy

            # Nếu muốn Gallery luôn có slot cho 4 ảnh, kể cả khi file không tồn tại (sẽ hiển thị lỗi ảnh)
            # thì dùng example_image_files trực tiếp, nhưng tốt hơn là kiểm tra file
            if gallery_value: # Chỉ hiển thị gallery nếu có ít nhất 1 ảnh
                gr.Gallery(
                    value=gallery_value,
                    label="4 Ảnh MRI Minh Họa",
                    columns=2,  # Hiển thị 2 ảnh trên một hàng (tổng 2 hàng) hoặc 4 để 4 ảnh trên 1 hàng
                    object_fit="contain", # hoặc "cover"
                    height="auto" # Tự động điều chỉnh chiều cao
                )
            else:
                gr.Markdown("<p style='color:orange;'>Không thể tải ảnh minh họa. Vui lòng kiểm tra đường dẫn và tên tệp.</p>")  
        with gr.Column(scale=1):
            gr.Markdown("### 💬 Trò chuyện với Chatbot Y Tế")
            # Ẩn vùng chat ban đầu, chỉ hiện sau khi phân tích ảnh
            with gr.Group(visible=False) as chat_interface_group:
                chatbot_display = gr.Chatbot(label="Chatbot", height=400)
                user_chat_input = gr.Textbox(label="Nhập câu hỏi của bạn:", placeholder="Hỏi tôi về thông tin chung liên quan đến kết quả...")
                send_button = gr.Button("✉️ Gửi")
    
    gr.Markdown(
        "<p style='text-align:center; color:red; font-weight:bold;'>Chatbot chỉ cung cấp thông tin tham khảo. Luôn hỏi ý kiến bác sĩ!</p>"
    )

    # Kết nối các hành động
    analyze_button.click(
        fn=handle_image_analysis_and_start_chat,
        inputs=[input_image, app_state],
        outputs=[output_prediction, chatbot_display, app_state, chat_interface_group]
    )

    # Xử lý khi người dùng gửi tin nhắn trong textbox (nhấn Enter)
    user_chat_input.submit(
        fn=handle_user_chat_message,
        inputs=[user_chat_input, chatbot_display, app_state],
        outputs=[chatbot_display, app_state]
    ).then(lambda: "", outputs=[user_chat_input]) # Xóa textbox sau khi gửi

    # Xử lý khi người dùng nhấn nút "Gửi"
    send_button.click(
        fn=handle_user_chat_message,
        inputs=[user_chat_input, chatbot_display, app_state],
        outputs=[chatbot_display, app_state]
    ).then(lambda: "", outputs=[user_chat_input]) # Xóa textbox sau khi gửi


if __name__ == "__main__":
    if keras_model is None:
        print("Không thể khởi chạy ứng dụng do model Keras chưa được tải.")
    elif gemini_model is None:
        print("Khởi chạy ứng dụng, nhưng chatbot Gemini sẽ không hoạt động do API key chưa được cấu hình hoặc có lỗi.")
        demo.launch()
    else:
        demo.launch()
