import 'package:flutter/material.dart';

void main() {
  runApp(new MaterialApp(
    home: new MagicDrawApp()
    )
  ); 
}

class MagicDrawApp extends StatelessWidget {
  @override
  Widget build(BuildContext context){
    return MaterialApp(
      home: Scaffold(
        body: Column(
          children: [
            Expanded(
              flex:9,
              child: Container(
                color: Colors.yellow[50],
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    Column(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      children: [
                      new Text("aaa      ")
                    ]),
                    Column(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      children: [
                        new Text("asdasda")
                      ],)
                  ]),
              )
            ),
            Expanded(
              flex:1,
              child: Container(
                color: Colors.blue,
                child: IconButton(
                  icon: Icon(Icons.arrow_forward_ios),
                ),
              ),
            ),
          ],)
      ),
    );
  }
}