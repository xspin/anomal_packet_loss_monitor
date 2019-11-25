var data_select //tunip list dict
var data // plot data
var data_setting


Date.prototype.fmt = function() {
  var mm = this.getMonth() + 1; // getMonth() is zero-based
  var dd = this.getDate();
  return [this.getFullYear(),
          (mm>9 ? '' : '0') + mm,
          (dd>9 ? '' : '0') + dd
         ].join('-');
};

function plot(data){
    var trace_overdrop = {
        name: 'Overlay 丢包率',
        x: data['Time'],
        y: data['Data'],
        type: 'scatter',
        line: {
            color: '#0D47A1',
            width: 1
        },
        xaxis: 'x',
        yaxis: 'y',
    };
    var trace_overdrop_warn = {
        name: 'Overlay 告警',
        x: data['Time'],
        y: data['Warn'],
        mode: 'scatter',
        line: {
            color: 'red',
            width: 2
        },
        xaxis: 'x',
        yaxis: 'y',
    }
    var trace_underdrop = {
        name: 'Underlay 丢包率',
        x: data['Time'],
        y: data['Data2'],
        type: 'scatter',
        line: {
            color: 'green',
            width: 1
        },
        xaxis: 'x',
        yaxis: 'y',
    };
    var trace_underdrop_warn = {
        name: 'Underlay 告警',
        x: data['Time'],
        y: data['Warn2'],
        mode: 'scatter',
        line: {
            color: '#F9A825',
            width: 2
        },
        xaxis: 'x',
        yaxis: 'y',
    }
    var trace_delay = {
        name: '时延',
        x: data['Time'],
        y: data['Delay'],
        type: 'scatter',
        line: {
            color: '#6A1B9A',
            width: 1
        },
        xaxis: 'x',
        yaxis: 'y2',
    };
    var traces = [
        trace_underdrop, trace_underdrop_warn, 
        trace_overdrop, trace_overdrop_warn,
        trace_delay, 
    ];
    ft = data['Time'][0]
    lt = data['Time'][data['Time'].length-1]
    t = new Date(lt)-3600*7*24*1000
    if (t > new Date(ft)) {
        ft = new Date(t).fmt();
    }
    var NSIDNAME = $('#nsid')[0].selectedOptions[0].text;
    var TUNIP = $('#tunip')[0].selectedOptions[0].text;
    var METHOD = $('#method')[0].selectedOptions[0].text;
    var UNDERLAY = "Underlay：" + data['Underlay'].join(', ')
    var layout = {
        height: 550,
        grid:{
            rows: 2,
            columns: 1,
            subplots: [['xy'],['xy2']],
            roworder: 'top to bottom'
        },
        title: NSIDNAME + ' ' + TUNIP + ' - ' + METHOD + '<br>' + UNDERLAY,
        xaxis: {
            range: [ft, lt],
            rangeslider: {
            },
            zerolinewidth: 0,
            type: 'date',
        },
        yaxis: {
            // fixedrange: false,
            range: [-10, 105],
            title: '丢包率（%）',
            type: 'linear'
        },
        yaxis2: {
            // fixedrange: false,
            range: 'auto',
            title: '时延（ms）',
            type: 'linear'
        },
    };
    Plotly.purge('graph')
    Plotly.newPlot('graph', traces, layout, {
        scrollZoom: false
    });
    $('.range').show();
    reset_ymax();
    $('#setting_panel').hide();
    // $('#underlay')[0].innerHTML = "<p>Underlay：" + data['Underlay'].join(', ') + "</p>";
}

function check_validation() {
    var nsid = $('#nsid')[0].selectedOptions[0].value;
    var tunip = $('#tunip')[0].selectedOptions[0].value;
    if (nsid=='0' || tunip=='0') {
        alert('请选择项目和链路！');
        return false;
    }
    return true;
}
function get_post_data() {
    var data_post = {};
    data_post['nsid'] = $('#nsid')[0].selectedOptions[0].value;
    data_post['tunip'] = $('#tunip')[0].selectedOptions[0].value;
    data_post['method'] = $('#method')[0].selectedOptions[0].value;
    data_post['setting'] = JSON.parse(JSON.stringify(data_setting[data_post['method']]));
    for (key in data_post['setting']) {
        data_post['setting'][key].pop();
    }
    return data_post;
}

var loading_timer_t = 0;
var timeoutVariable;
function loading(flag) {
    if (flag) {
        $('#loading').show();
        loading_timer_t = 0;
        function myTimer() {
            loading_timer_t += 1;
            $('#timer')[0].innerHTML = '<p>耗时：' + loading_timer_t + 's</p>';
        }
        timeoutVariable = setInterval(function(){myTimer()},1000);
    } else {
        $('#loading').hide();
        window.clearInterval(timeoutVariable);
        $('#timer')[0].innerHTML = '';
    }
}

function select_submit() {  
    if (!check_validation()) return false;
    loading(true);
    $.ajax({  
            type: "POST",  
            url: "select",  
            // data: $('#form_select').serialize(),// 你的formid  
            data: JSON.stringify(get_post_data()),
            // async: true,  
            error: function(xhr, textStatus, error) {  
                alert('请求失败！');
                loading(false);
            },  
            success: function(dat) {  
                try {
                    data = JSON.parse(dat);
                } catch (error) {
                    alert('请求失败！ MSG:' + dat);
                    loading(false);
                    return ;
                }
                try {
                    plot(data);
                } catch (error) {
                    alert('画图失败！ Error:'+error);
                }
                loading(false);
            }  
        }
        );  
    return false;
}  

function init() {  
    loading(true);
    $.ajax({  
            type: "POST",  
            url: "getlist",  
            error: function(xhr, textStatus, error) {  
                console.log('error: ' + xhr.error)
                console.log(error);
                alert('加载失败！');
                loading(false);
            },  
            success: function(data) {  
                try {
                    data_select = JSON.parse(data);
                    set_nsid();
                } catch (error) {
                    alert('加载失败！MSG:'+error);
                }
                loading(false);
            }  
        }
        );  
        $.getJSON("getsetting", function(json){
            // console.log(json);
            data_setting = json;
        });
}  
function set_nsid() {
    nsname = data_select['nsname']
    $('#nsid').empty();
    $('#nsid').append('<option value="0" style="color:gray">选择项目</option>');
    var highlight_nsid = [500850, 504608, 502902]
    for(var i in highlight_nsid) {
        var nsid = highlight_nsid[i];
        option = "<option style='color:red' value='" + nsid + "'>" + nsid+" "+nsname[nsid] + "</option>";
        $('#nsid').append(option);
    }
    for(var nsid in nsname) {
        option = "<option value='" + nsid + "'>" + nsid+" "+nsname[nsid] + "</option>";
        $('#nsid').append(option);
    }
}
function set_tunip() {
    nsid = $('#nsid')[0].selectedOptions[0].value;
    nsip = data_select['nsip']
    $('#tunip').empty();
    $('#tunip').append('<option value="0" style="color:gray">选择链路</option>');
    for(var i in nsip[nsid]) {
        option = "<option value='" + nsip[nsid][i] + "'>" + nsip[nsid][i] + "</option>";
        $('#tunip').append(option);
    }
}
function reset_ymax() {
    $('#ymax')[0].value = 100;
    $('#ymax_value')[0].innerHTML = '100%';
}
function change_ymax() {
    var ymax = $('#ymax')[0].value;
    $('#ymax_value')[0].innerHTML = ymax+'%';
    var update = {
        'yaxis.range': [-ymax/10, ymax*1.05]
    }
    Plotly.relayout(graph, update);
}
function change_delay_ymax() {
    var ymax = $('#delay_ymax')[0].value;
    $('#delay_ymax_value')[0].innerHTML = ymax+' ms';
    var update = {
        'yaxis2.range': [0, ymax]
    }
    Plotly.relayout(graph, update);
}
function setting() {
    if($('#setting_panel')[0].style.display=='none' || $('#setting_panel')[0].style.display=="") {
        $('#setting_panel').show()
        draw_setting_panel(data_setting);
    } else {
        $('#setting_panel').hide()
    }
}
function draw_setting_panel(json) {
    var method = $('#method')[0].selectedOptions[0].value;
    $('#ul_setting').empty();
    for(v in json[method]) {
        var li = '<li><span>' + v + '</span>: <input type="text" style="width:50px" value="'+ json[method][v][0]+'"> ' + json[method][v][1] + '</li>';
        $('#ul_setting').append(li)
    }
}
function setting_submit() {
    param_input = $('#ul_setting li input')
    param_name = $('#ul_setting li span')
    var method = $('#method')[0].selectedOptions[0].value;
    for(var i=0; i<param_input.length; i++) {
        name = param_name[i].innerHTML;
        value = param_input[i].value;
        // console.log( name+': '+value);
        data_setting[method][name][0] = value;
    }
    return select_submit();
}