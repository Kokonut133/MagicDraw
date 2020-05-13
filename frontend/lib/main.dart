import 'dart:convert';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';

void main() {
  runApp(new MaterialApp(
    home: new MagicDrawApp()
    )
  ); 
}

const canvas_width = 400.0;
const canvas_height = 400.0;

// MediaQuery.of(context).size.height / 2;

class MagicDrawApp extends StatelessWidget {
  @override
  Widget build(BuildContext context){
    return MaterialApp(
      home: Scaffold(
        body: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Expanded(
              flex:9,
              child: Container(
                //color: Colors.yellow[50],
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    Expanded(
                      flex: 5,
                      child: Padding(
                        padding: EdgeInsets.all(20),
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                          children: [
                            FittedBox(
                              child: SizedBox(
                                child: ClipRect(
                                  child:My_custom_painter_page(),
                                )
                              ),
                            )]
                        ),
                      )
                    ),
                    Expanded(
                      flex: 5,
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                        children: [
                          Real_image_generator()]
                      )
                    )]
                ),
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

class My_custom_painter_page extends StatefulWidget{
  My_custom_painter_page({Key key}): super(key: key);

  @override
  State<StatefulWidget> createState() => _My_custom_painter_page();
}

class _My_custom_painter_page extends State<My_custom_painter_page> {
  List<Offset> _offsets = [];
  Color activeColor;
  List<Color> _colors = [];
  List<double> _brushSizes = [];
  double brushSize = 4;
  void onChangeColor(Color value) {
    activeColor = value;
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onPanUpdate: (details) {
        setState(() {
          RenderBox _object = context.findRenderObject();
          Offset _locationPoints =
              _object.globalToLocal(details.globalPosition);

          _offsets.add(_locationPoints);
          _colors.add(activeColor);
          _brushSizes.add(brushSize);
        });
      },
      onPanEnd: (details) {
        _offsets.add(null);
        _colors.add(null);
        _brushSizes.add(null);
      },
      child: Stack(
        children: <Widget>[
          Container(
            decoration: BoxDecoration(
              border: Border.all(),
              color: Colors.white),
            width: canvas_width,
            height: canvas_height,
          ),
          CustomPaint(
            painter: My_custom_painter(
                offsets: _offsets, colors: _colors, brushSizes: _brushSizes),
            size: Size.square(MediaQuery.of(context).size.width / 2)
          ),
        ],
      )
    );
  }
}

class My_custom_painter extends CustomPainter {
  final List<Offset> offsets;
  final List<Color> colors;
  final List<double> brushSizes;
  final brush = Paint()
    ..strokeCap = StrokeCap.round
    ..strokeWidth = 4.0
    ..color = Colors.red
    ..isAntiAlias = true;

  My_custom_painter(
      {@required this.offsets,
      @required this.colors,
      @required this.brushSizes});

  @override
  void paint(Canvas canvas, Size size) {
    for (int i = 0; i < offsets.length - 1; i++) {
      if (offsets[i] != null && offsets[i + 1] != null) {
        brush.color = colors[i] == null ? Colors.red : colors[i];
        brush.strokeWidth = brushSizes[i];
        canvas.drawLine(offsets[i], offsets[i + 1], brush);
      }
    }
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) {
    return true;
  }
}

class Real_image_generator extends StatefulWidget {
  @override
  State<StatefulWidget> createState() => _Real_image_generator_state();
  }
  
class _Real_image_generator_state extends State<Real_image_generator>{
  ByteData imgBytes;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize:  MainAxisSize.max,
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: <Widget>[
          imgBytes != null ? Center(
            child: Image.memory(
              Uint8List.view(imgBytes.buffer),
              width: canvas_width,
              height: canvas_height,
            ))
          : Container(child:Text("asdasdas")),
          Padding(
            padding: const EdgeInsets.all(12.0),
            child: RaisedButton(
                child: Text('Generate image'), onPressed: generate_image),
          ),
        ]
      )
    );
  }

  void generate_image() async {
    final color = Colors.pink;
    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder,
      Rect.fromPoints(Offset(0.0,0.0), Offset(canvas_width, canvas_height)));
    
    final stroke = Paint()
      ..color = Colors.grey
      ..style = PaintingStyle.stroke;
    canvas.drawRect(Rect.fromLTWH(0.0, 0.0, 50, 50), stroke);

    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;
    canvas.drawCircle(Offset(50, 50), 20.0, paint);

    final picture = recorder.endRecording();
    final img = await picture.toImage(200, 200);
    final pngBytes = await img.toByteData(format: ImageByteFormat.png);

    setState(() {
      imgBytes = pngBytes;
    });
  }
  
}