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
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Expanded(
              flex:9,
              child: Container(
                color: Colors.yellow[50],
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
                                child: My_custom_painter_page(),
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
                          new Text("asdassdfsdfda")
                        ],)
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
                  onPressed: () {  },
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
            width: MediaQuery.of(context).size.width / 2,
            height: MediaQuery.of(context).size.height / 2,
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