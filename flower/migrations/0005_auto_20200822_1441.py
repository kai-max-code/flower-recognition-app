# Generated by Django 3.1 on 2020-08-22 14:41

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('flower', '0004_auto_20200821_1415'),
    ]

    operations = [
        migrations.AlterField(
            model_name='image',
            name='fifth_value',
            field=models.CharField(default='', max_length=10),
        ),
        migrations.AlterField(
            model_name='image',
            name='first_value',
            field=models.CharField(default='', max_length=10),
        ),
        migrations.AlterField(
            model_name='image',
            name='fourth_value',
            field=models.CharField(default='', max_length=10),
        ),
        migrations.AlterField(
            model_name='image',
            name='second_value',
            field=models.CharField(default='', max_length=10),
        ),
        migrations.AlterField(
            model_name='image',
            name='third_value',
            field=models.CharField(default='', max_length=10),
        ),
    ]
