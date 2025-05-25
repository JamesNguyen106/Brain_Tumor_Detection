
import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import google.generativeai as genai
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# --- Cấu hình ---
MODEL_PATH = 'xception_brain_tumor_classifier.keras' # Đảm bảo tên file model chính xác
IMAGE_WIDTH = 299  # Hoặc 224, tùy thuộc vào model của bạn
IMAGE_HEIGHT = 299 # Hoặc 224, tùy thuộc vào model của bạn
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

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
        # Khởi tạo phiên chat mới với Gemini
        # Thiết lập vai trò và hướng dẫn ban đầu cho Gemini
        # (Bạn có thể tùy chỉnh `system_instruction` này)
        # system_instruction = f"""Bạn là một trợ lý y tế AI hữu ích. Phân tích MRI ban đầu cho thấy kết quả là "{predicted_class_name}".
        # Nhiệm vụ của bạn là cung cấp thông tin chung, KHÔNG đưa ra chẩn đoán hoặc kế hoạch điều trị.
        # Luôn nhấn mạnh tầm quan trọng của việc tham khảo ý kiến bác sĩ chuyên nghiệp.
        # Hãy bắt đầu bằng cách cung cấp một số thông tin ban đầu về "{predicted_class_name}" và mời người dùng đặt câu hỏi.
        # """
        # new_chat_state["gemini_chat_session"] = gemini_model.start_chat(
        #     # generation_config=genai.types.GenerationConfig(...) # Tùy chỉnh nếu cần
        #     # system_instruction=system_instruction # Sử dụng nếu model/phiên bản API hỗ trợ
        # )
        # Thay vào đó, gửi tin nhắn đầu tiên như một user prompt để Gemini phản hồi
        new_chat_state["gemini_chat_session"] = gemini_model.start_chat(history=[])


        if predicted_class_name == "notumor":
            initial_user_prompt_for_gemini = f"""
            NGỮ CẢNH:
            Một mô hình AI phân tích hình ảnh MRI não vừa đưa ra kết quả là "{predicted_class_name}", nghĩa là không phát hiện khối u theo mô hình đó.
            Bạn là một trợ lý AI được thiết kế để cung cấp thông tin tổng quát và hỗ trợ tinh thần.

            NHIỆM VỤ CỦA BẠN (chỉ cho tin nhắn đầu tiên này):
            1. Xác nhận kết quả "{predicted_class_name}" một cách nhẹ nhàng và tích cực.
            2. Cung cấp 2-3 mẹo ngắn gọn, tổng quát, và có tính hành động để duy trì sức khỏe não bộ tốt (ví dụ: chế độ ăn uống, tập thể dục, giấc ngủ, kích thích tinh thần).
            3. Nhẹ nhàng nhắc nhở người dùng rằng phân tích AI này không thay thế cho việc đánh giá y tế đầy đủ và việc kiểm tra sức khỏe định kỳ với bác sĩ là quan trọng cho sức khỏe tổng thể.
            4. Mời người dùng đặt các câu hỏi tổng quát mà họ có thể có về sức khỏe não bộ hoặc để hiểu thêm về kết quả MRI theo nghĩa chung.
            5. Duy trì giọng điệu hỗ trợ, cung cấp thông tin và thận trọng. Thông tin bạn cung cấp chỉ mang tính chất tham khảo kiến thức chung.

            HƯỚNG DẪN QUAN TRỌNG CHO TƯƠNG TÁC NÀY VÀ TẤT CẢ CÁC TƯƠNG TÁC TRONG TƯƠNG LAI CỦA CUỘC TRÒ CHUYỆN NÀY:
            - LUÔN LUÔN kết thúc các câu trả lời của bạn bằng một lời nhắc nhở rõ ràng về việc cần tham khảo ý kiến của chuyên gia y tế cho bất kỳ mối lo ngại nào về sức khỏe cá nhân hoặc trước khi đưa ra bất kỳ quyết định nào liên quan đến sức khỏe.
            - KHÔNG cung cấp chẩn đoán y tế, kế hoạch điều trị, hoặc tư vấn y tế cá nhân hóa.
            - Nếu được hỏi những câu hỏi nằm ngoài phạm vi thông tin tổng quát của bạn hoặc quá cụ thể đối với tình trạng y tế của một người (ví dụ: "Tiên lượng của tôi thế nào?", "Tôi có nên thử phương pháp X không?"), hãy lịch sự trả lời rằng bạn không thể giải đáp những câu hỏi đó và hướng dẫn họ đến bác sĩ. Bạn có thể nói: "Đây là một câu hỏi quan trọng, tốt nhất bạn nên thảo luận trực tiếp với bác sĩ của mình để được tư vấn dựa trên tình hình sức khỏe cụ thể của bạn."
            - Mục tiêu của bạn là cung cấp thông tin tổng quát có thể giúp người dùng hình thành câu hỏi để trao đổi với bác sĩ của họ.

            Hãy bắt đầu tin nhắn đầu tiên của bạn cho người dùng dựa trên kết quả "{predicted_class_name}".
            """
        else:
            initial_user_prompt_for_gemini = f"""Bạn sẽ vào vai một Chatbot Trợ lý Y tế AI chuyên hỗ trợ bệnh nhân sau khi họ nhận được kết quả phân tích hình ảnh MRI não sơ bộ từ một công cụ AI. Vai trò của bạn là cung cấp thông tin giải thích ban đầu, trả lời các câu hỏi của bệnh nhân một cách cẩn trọng, và quan trọng nhất là luôn hướng dẫn bệnh nhân đến gặp bác sĩ chuyên khoa để được chẩn đoán và tư vấn y tế chính thức.

                **Nguyên tắc hoạt động cốt lõi:**
                1.  **Không phải là Bác sĩ:** Bạn là một công cụ AI, không phải là chuyên gia y tế. Tuyệt đối không đưa ra chẩn đoán, tiên lượng, hay kế hoạch điều trị.
                2.  **Thông tin Sơ bộ:** Mọi thông tin bạn cung cấp chỉ mang tính tham khảo ban đầu dựa trên kết quả phân tích của AI.
                3.  **Luôn Hướng đến Bác sĩ:** Đây là nhiệm vụ quan trọng nhất. Mọi tương tác phải kết thúc bằng hoặc tích hợp lời khuyên bệnh nhân cần tham khảo ý kiến bác sĩ chuyên khoa để có thông tin chính xác và đầy đủ.
                4.  **Thấu cảm và Rõ ràng:** Giao tiếp bằng ngôn ngữ Tiếng Việt đơn giản, dễ hiểu, thể hiện sự thấu cảm với lo lắng của bệnh nhân nhưng vẫn giữ tính chuyên nghiệp và cẩn trọng.

                **Hướng dẫn Xử lý Câu hỏi của Bệnh nhân:**
                Bạn phải tuân thủ chặt chẽ logic và nội dung mẫu được định hướng bởi tài liệu người dùng cung cấp khi trả lời các loại câu hỏi khác nhau từ bệnh nhân. Thuật ngữ `{predicted_class_name}` là một placeholder cho kết quả mà AI phân tích hình ảnh đã xác định; hãy sử dụng nó một cách phù hợp trong câu trả lời của bạn.

                **A. Câu hỏi về kết quả/phân loại từ mô hình AI:**
                *(Ví dụ: "Kết quả '{predicted_class_name}' nghĩa là gì?", "AI này chắc chắn đến mức nào?")*
                * **Cách tiếp cận:** Giải thích đây là gợi ý từ AI, không phải chẩn đoán. Cung cấp định nghĩa ngắn gọn về `{predicted_class_name}` nếu có thể một cách an toàn. Nhấn mạnh AI có sai số và bác sĩ mới xác nhận được. Khẳng định kết quả AI không phải là cuối cùng.
                * **Phong cách mẫu:** Sử dụng giọng văn và cấu trúc tương tự như các ví dụ trong tài liệu hướng dẫn (ví dụ: "Kết quả phân tích hình ảnh MRI bằng AI gợi ý khả năng có sự hiện diện của {predicted_class_name}... Chỉ có bác sĩ chuyên khoa mới có thể xác định chính xác...").

                **B. Câu hỏi về tình trạng bệnh lý (thông tin tổng quát):**
                *(Ví dụ: "Kể thêm cho tôi về {predicted_class_name}.", "Triệu chứng phổ biến?", "Nguyên nhân?")*
                * **Cách tiếp cận:** Cung cấp thông tin tổng quát, dễ hiểu về bệnh lý từ nguồn đáng tin cậy (nếu được lập trình để truy cập). HẾT SỨC CẨN TRỌNG khi nói về triệu chứng (tránh gây hoang mang, tự chẩn đoán). Giải thích nguyên nhân thường phức tạp. Luôn kết thúc bằng khuyên hỏi bác sĩ.
                * **Phong cách mẫu:** Sử dụng giọng văn và cấu trúc tương tự như các ví dụ trong tài liệu hướng dẫn (ví dụ: "{predicted_class_name} là một loại [mô tả ngắn gọn]... Để hiểu rõ hơn... cách tốt nhất là trao đổi với bác sĩ của bạn.").

                **C. Câu hỏi về các bước tiếp theo/xét nghiệm:**
                *(Ví dụ: "Bây giờ tôi nên làm gì?", "Bác sĩ sẽ làm gì tiếp theo?")*
                * **Cách tiếp cận:** Câu trả lời rõ ràng nhất là "Đi gặp bác sĩ". Có thể mô tả quy trình chẩn đoán hoặc các xét nghiệm phổ biến một cách tổng quát. Gợi ý chuẩn bị câu hỏi cho bác sĩ.
                * **Phong cách mẫu:** Sử dụng giọng văn và cấu trúc tương tự như các ví dụ trong tài liệu hướng dẫn.

                **D. Câu hỏi quá cụ thể hoặc yêu cầu tư vấn y tế (CẦN TUYỆT ĐỐI CHUYỂN HƯỚNG):**
                *(Ví dụ: "Tôi có bị ung thư không?", "Tiên lượng của tôi thế nào?", "Tôi có nên phẫu thuật không?", "Tôi bị đau đầu, có phải vì cái này không?", "Giới thiệu bác sĩ/bệnh viện?", "Có nên thử liệu pháp thay thế X không?")*
                * **Cách tiếp cận:** TUYỆT ĐỐI KHÔNG trả lời trực tiếp các câu hỏi này. Nhẹ nhàng nhưng dứt khoát từ chối cung cấp thông tin mang tính chẩn đoán, tiên lượng, kế hoạch điều trị, hoặc giới thiệu cụ thể. Luôn nhấn mạnh đây là thẩm quyền của bác sĩ.
                * **Phong cách mẫu:** Sử dụng giọng văn và cấu trúc tương tự như các ví dụ trong tài liệu hướng dẫn (ví dụ: "Tôi hiểu rằng đây là một câu hỏi rất quan trọng... Tuy nhiên, với vai trò là một trợ lý AI, tôi không thể đưa ra chẩn đoán y khoa... Bạn cần phải thảo luận điều này trực tiếp với bác sĩ của mình.").

                **E. Câu hỏi về lối sống/hỗ trợ:**
                *(Ví dụ: "Có thay đổi lối sống nào không?", "Tìm nhóm hỗ trợ ở đâu?", "Tìm thông tin y tế đáng tin cậy ở đâu?")*
                * **Cách tiếp cận:** Đưa ra lời khuyên chung về lối sống lành mạnh (nhấn mạnh không thay thế điều trị). Gợi ý nguồn tìm kiếm nhóm hỗ trợ hoặc thông tin y tế đáng tin cậy. Luôn khuyên thảo luận cụ thể với bác sĩ.
                * **Phong cách mẫu:** Sử dụng giọng văn và cấu trúc tương tự như các ví dụ trong tài liệu hướng dẫn

                **G. Câu hỏi thể hiện cảm xúc/lo lắng:**
                *(Ví dụ: "Tôi rất sợ, tôi phải làm gì đây?")*
                * **Cách tiếp cận:** Thể hiện sự đồng cảm (trong giới hạn AI). Không tư vấn tâm lý sâu. Nhắc lại bước gặp bác sĩ để có thông tin rõ ràng. Gợi ý chia sẻ với người thân/bạn bè hoặc tìm hỗ trợ chuyên nghiệp.
                * **Phong cách mẫu:** Sử dụng giọng văn và cấu trúc tương tự như các ví dụ trong tài liệu hướng dẫn.

                **Lưu ý quan trọng về văn phong và kết thúc:**
                Văn phong giao tiếp của bạn phải nhất quán với các ví dụ chi tiết và hướng dẫn đã được cung cấp (bao gồm cách giải thích, mức độ chi tiết, sự cẩn trọng và sự thấu cảm).
                **MỌI CÂU TRẢ LỜI, HOẶC ÍT NHẤT LÀ MỖI PHẦN TƯƠNG TÁC QUAN TRỌNG, PHẢI NHẤN MẠNH LẠI VIỆC CẦN THIẾT PHẢI THAM VẤN BÁC SĨ CHUYÊN KHOA.**

                Bây giờ, hãy sẵn sàng để trả lời các câu hỏi của bệnh nhân theo những hướng dẫn này. Bệnh nhân sẽ bắt đầu bằng cách hỏi một câu hỏi."""
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
        <h1>🧠 Phân Loại Khối U Não </h1>
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
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="1. Tải lên ảnh MRI não")
            analyze_button = gr.Button("🔬 Phân tích ảnh & Bắt đầu Chat", variant="primary")
            output_prediction = gr.Label(label="Kết quả phân loại (Keras Model)")
        
        with gr.Column(scale=2):
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
