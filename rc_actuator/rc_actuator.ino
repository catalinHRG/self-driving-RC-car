int pin = 8;
int data = 0;

void setup() {

  Serial.begin(115200);
  pinMode(8, OUTPUT);
  pinMode(9, OUTPUT);
  pinMode(10, OUTPUT);

  digitalWrite(8, LOW);
  
}

void loop() {

 
  if(Serial.available() > 0){
      pin = pin + 1;

      if(pin == 11) { pin = 8; }
    
  }

}
