#include <WiFi.h>  // ✅ 新增 Wi-Fi 库
#include <magic_wand_inferencing.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// ==== Wi-Fi 设置 ====
const char* ssid = "UW MPSK";
const char* password = "Y(:3J$$yeE";  // ← 保持与你注册后得到的密码完全一致
    // ⚠️替换为你的热点密码

// ==== Pin Definitions ====
#define LED_PIN 43  // D6
#define SAMPLE_RATE_MS 10
#define CAPTURE_DURATION_MS 5000
#define INFERENCE_INTERVAL_MS 3000
#define CONFIDENCE_THRESHOLD 80.0

// ==== Global Variables ====
Adafruit_MPU6050 mpu;
float features[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];
bool capturing = false;
unsigned long last_sample_time = 0;
unsigned long capture_start_time = 0;
unsigned long last_inference_time = 0;
int sample_count = 0;
String serverUrl = "http://172.20.10.9:8000/predict"; // ← 你的服务器地址

// ==== Wi-Fi 连接函数 ====
void connectToWiFi() {
  Serial.print("Connecting to ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\n✅ WiFi connected!");
  Serial.print("ESP32 IP address: ");
  Serial.println(WiFi.localIP());
}

// ==== Setup ====
void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  digitalWrite(LED_PIN, HIGH);
  delay(200);
  digitalWrite(LED_PIN, LOW);

  connectToWiFi();  // ✅ 新增连接 Wi-Fi

  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1) delay(10);
  }

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  Serial.println("MPU6050 ready");

  last_inference_time = millis();
}

void blinkLED(int times) {
  for (int i = 0; i < times; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(200);
    digitalWrite(LED_PIN, LOW);
    delay(200);
  }
}

int raw_feature_get_data(size_t offset, size_t length, float *out_ptr) {
  memcpy(out_ptr, features + offset, length * sizeof(float));
  return 0;
}

void sendRawDataToServer() {
  HTTPClient http;
  http.begin(serverUrl);
  http.addHeader("Content-Type", "application/json");

  String jsonPayload = "{\"input\":[";
  for (int i = 0; i < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE; i++) {
    jsonPayload += String(features[i], 6);
    if (i < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE - 1) jsonPayload += ",";
  }
  jsonPayload += "]}";

  int httpResponseCode = http.POST(jsonPayload);
  Serial.print("HTTP Response code: ");
  Serial.println(httpResponseCode);

  if (httpResponseCode > 0) {
    String response = http.getString();
    Serial.println("Server response: " + response);

    DynamicJsonDocument doc(512);
    DeserializationError error = deserializeJson(doc, response);
    if (!error) {
      int prediction = doc["prediction"];
      float confidence = doc["raw_output"][0][prediction];

      Serial.print("Predicted gesture: ");
      Serial.println(prediction);
      Serial.print("Confidence: ");
      Serial.print(confidence * 100, 1);
      Serial.println("%");

      // 可加LED动作
    } else {
      Serial.print("JSON parse error: ");
      Serial.println(error.c_str());
    }
  } else {
    Serial.println("POST failed.");
  }

  http.end();
}

void run_inference() {
  signal_t signal;
  signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
  signal.get_data = &raw_feature_get_data;

  ei_impulse_result_t result = { 0 };
  EI_IMPULSE_ERROR res = run_classifier(&signal, &result, false);

  if (res != EI_IMPULSE_OK) {
    Serial.println("Classifier failed");
    return;
  }

  float max_val = 0;
  const char* label = "";
  for (int i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
    if (result.classification[i].value > max_val) {
      max_val = result.classification[i].value;
      label = result.classification[i].label;
    }
  }

  Serial.print("Detected: ");
  Serial.print(label);
  Serial.print(" (");
  Serial.print(max_val * 100, 1);
  Serial.println("%)");

  if (max_val * 100 < CONFIDENCE_THRESHOLD) {
    Serial.println("Low confidence – sending to server...");
    sendRawDataToServer();
  } else {
    if (strcmp(label, "O") == 0) {
      blinkLED(6);
    } else if (strcmp(label, "V") == 0) {
      blinkLED(3);
    } else if (strcmp(label, "A") == 0) {
      blinkLED(1);
    }
  }
}

void capture_data() {
  if (millis() - last_sample_time >= SAMPLE_RATE_MS) {
    last_sample_time = millis();

    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    if (sample_count < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE / 3) {
      int i = sample_count * 3;
      features[i]     = a.acceleration.x;
      features[i + 1] = a.acceleration.y;
      features[i + 2] = a.acceleration.z;
      sample_count++;
    }

    if (millis() - capture_start_time >= CAPTURE_DURATION_MS) {
      capturing = false;
      run_inference();
    }
  }
}

void loop() {
  unsigned long now = millis();
  if (!capturing && (now - last_inference_time >= INFERENCE_INTERVAL_MS)) {
    Serial.println("Starting automatic inference...");
    sample_count = 0;
    capturing = true;
    capture_start_time = now;
    last_sample_time = now;
    last_inference_time = now;
  }
  if (capturing) {
    capture_data();
  }
}
