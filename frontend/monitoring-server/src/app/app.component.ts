import { Component, OnInit, OnDestroy, SecurityContext } from '@angular/core';
import { Observable, Subscription } from 'rxjs';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';


import {
  IMqttMessage,
  MqttService,
} from 'ngx-mqtt';

export interface RegisteredDevice {
  name: string;
  activeDetections: number;
  detectionImg?: SafeResourceUrl;
  backgroundColor: string;
  deterState?: string;
}

export interface DetectionHistory {
    name: string;
    lastSeen: string;
    travelDirection: string;
    lastSeenTime: Date;
}

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit, OnDestroy {
  private subscription: Subscription;
  private detSubscription: Subscription;
  REGISTRATION: string = 'frontend/registration';
  DETECTION_HISTORY: string = 'frontend/detectionHistory';
  REFRESH = 'refresh';

  title = 'monitoring-server';
  devices: RegisteredDevice[] = [
  ];

  detectionHistory: DetectionHistory[] = [
  ]; 

  regChannels = new Map<string, Subscription>();

  constructor(private _mqttService: MqttService, private sanitizer: DomSanitizer) {
    this.subscription = this._mqttService.observe(this.REGISTRATION).subscribe(this.addDevice.bind(this));
    this.detSubscription = this._mqttService.observe(this.DETECTION_HISTORY).subscribe(this.setDetectionHistory.bind(this));
  }


  addDevice(message: IMqttMessage): void {
    let msg = message.payload.toString();

    // Add a subscription for device
    if (this.regChannels.has(msg)) {
      console.log('Device ' + msg + ' already registered!');
    } else {
      this.devices.push({ name: msg, activeDetections: 0, backgroundColor: 'white' });
      let dataChannel = 'frontend/' + msg;
      console.log('Subscribing to ' + dataChannel);
      this.regChannels.set(msg, this._mqttService.observe(dataChannel).subscribe(this.detectionReceived.bind(this)));
      let imageChannel = 'frontend/' + msg + '/image';
      console.log('Subscribing to ' + imageChannel);
      this.regChannels.set(msg, this._mqttService.observe(imageChannel).subscribe(this.detectionImageReceived.bind(this)));
    }
  }

  detectionReceived(message: IMqttMessage): void {
    let channel = message.topic.toString();
    let msg = message.payload.toString();
    try {
      let jsMsg = JSON.parse(msg);
      let devName: string = channel.split('/')[1];
      for (let device of this.devices) {
        if (device.name === devName) {
          device.activeDetections = jsMsg.activeDetections;
          if (jsMsg.deterState != null) {
            if (jsMsg.deterState == 'alert') {
              device.backgroundColor = 'yellow';
            } else if (jsMsg.deterState == 'success') {
              device.backgroundColor = 'green';
            } else if (jsMsg.deterState == 'failed') {
              device.backgroundColor = 'red';
            }
          } else {
            device.backgroundColor = "white";
          }
        }
      }
    } catch(e) {
      console.log('Invalid json received, ignoring...');
    }
     
  }

  detectionImageReceived(message: IMqttMessage): void {
    let channel = message.topic.toString();
    let devName = channel.split('/')[1];
    console.log('Received image...');
    for (let device of this.devices) {
      if (device.name === devName) {
        console.log('Updating image for device ' + devName);
        let url = 'data:image/jpg;base64,' + message.payload.toString(); 
        device.detectionImg = this.sanitizer.bypassSecurityTrustResourceUrl(url);
      }
    }
  }

  setDetectionHistory(message: IMqttMessage): void {
    console.log(JSON.parse(message.payload.toString()));
    this.detectionHistory = JSON.parse(message.payload.toString());
  }

  callRefresh(): void {
    this._mqttService.unsafePublish(this.REFRESH, '');
  }

  ngOnInit() :void {
    setTimeout(this.callRefresh.bind(this), 500); // Call after 1s
  }

  ngOnDestroy(): void {
    this.regChannels.forEach((value, key) => {
      value.unsubscribe();
    });
    this.detSubscription.unsubscribe();
    this.subscription.unsubscribe();
  }

}
